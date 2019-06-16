import argparse
import csv
import json
import os
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime

import numpy as np
import requests
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tqdm import tqdm, trange

# if in Google Colaboratory
try:
    from google.colab import drive
except:
    pass

from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer


def download_gpt2(model_name='117M'):
    """Downloads the GPT-2 model into the current directory
    from Google Cloud Storage.

    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    """

    subdir = os.path.join('models', model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                     'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                     'model.ckpt.meta', 'vocab.bpe']:

        r = requests.get("https://storage.googleapis.com/gpt-2/" +
                         subdir + "/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename,
                      total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def start_tf_sess(threads=-1):
    """
    Returns a tf.Session w/ config
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    return tf.Session(config=config)


def finetune(sess,
             dataset,
             steps=-1,
             model_name='117M',
             combine=50000,
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             restore_from='latest',
             run_name='run1',
             sample_every=100,
             sample_length=1023,
             sample_num=1,
             save_every=1000,
             print_every=1,
             max_checkpoints=1,
             use_memory_saving_gradients=False,
             only_train_transformer_layers=False,
             overwrite=False):
    """Finetunes the model on the given dataset.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        if file not in files:
            try:
                shutil.copyfile(os.path.join('models', model_name, file),
                                os.path.join(checkpoint_path, file))
            except FileNotFoundError as fnf_error:
                print("You need to download the GPT-2 model first via download_gpt2()")
                raise (fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if model_name != '117M':
        use_memory_saving_gradients = True
        only_train_transformer_layers = True
        accumulate_gradients = 1

    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=sample_length,
        context=context,
        batch_size=batch_size,
        temperature=1.0,
        top_p=0.9)

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars
    if accumulate_gradients > 1:
        if use_memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=tf.train.AdamOptimizer(learning_rate=learning_rate),
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if use_memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_log = tf.summary.FileWriter(checkpoint_path)

    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join('models', model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter - 1))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter - 1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter - 1) + '\n')

    def generate_samples():
        context_tokens = data_sampler.sample(1)
        all_text = []
        index = 0
        while index < sample_num:
            out = sess.run(
                tf_sample,
                feed_dict={context: batch_size * [context_tokens]})
            for i in range(min(sample_num - index, batch_size)):
                text = enc.decode(out[i])
                text = '======== SAMPLE {} ========\n{}\n'.format(
                    index + 1, text)
                all_text.append(text)
                index += 1
        print(text)
        maketree(os.path.join(SAMPLE_DIR, run_name))
        with open(
                os.path.join(SAMPLE_DIR, run_name,
                             'samples-{}').format(counter), 'w', encoding='utf8') as fp:
            fp.write('\n'.join(all_text))

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(batch_size)]

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))
        save()

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                save()
                return
            if (counter - 1) % save_every == 0 and counter > 1:
                save()
            if (counter - 1) % sample_every == 0 and counter > 1:
                generate_samples()

            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summary_loss),
                    feed_dict={context: sample_batch()})

            summary_log.add_summary(v_summary, counter)

            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                        .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')
        save()


def finetune_lr_cycle(sess,
                      dataset,
                      steps=-1,
                      model_name='117M',
                      combine=50000,
                      batch_size=1,
                      base_lr=0.00151,
                      max_lr=0.00906,
                      cycle_steps=10000,
                      accumulate_gradients=5,
                      restore_from='latest',
                      run_name='run1',
                      sample_every=100,
                      sample_length=1023,
                      sample_num=1,
                      save_every=1000,
                      print_every=1,
                      max_checkpoints=1,
                      use_memory_saving_gradients=False,
                      only_train_transformer_layers=False,
                      overwrite=False):
    """Finetunes the model on the given dataset using a cyclical lr

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        if file not in files:
            try:
                shutil.copyfile(os.path.join('models', model_name, file),
                                os.path.join(checkpoint_path, file))
            except FileNotFoundError as fnf_error:
                print("You need to download the GPT-2 model first via download_gpt2()")
                raise (fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if model_name != '117M':
        use_memory_saving_gradients = True
        only_train_transformer_layers = True
        accumulate_gradients = 1

    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=sample_length,
        context=context,
        batch_size=batch_size,
        temperature=1.0,
        top_p=0.9)

    global_step = tf.Variable(0, trainable=False)
    current_step = 0

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars
    if accumulate_gradients > 1:
        if use_memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=tf.train.AdamOptimizer(
                learning_rate=cyclic_learning_rate(global_step=global_step, step_size=cycle_steps / 2,
                                                   learning_rate=base_lr, max_lr=max_lr, mode='triangular2')),
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=cyclic_learning_rate(global_step=global_step, step_size=cycle_steps / 2,
                                               learning_rate=base_lr, max_lr=max_lr, mode='triangular2'))
        if use_memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_log = tf.summary.FileWriter(checkpoint_path)

    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join('models', model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter - 1))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter - 1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter - 1) + '\n')

    def generate_samples():
        context_tokens = data_sampler.sample(1)
        all_text = []
        index = 0
        while index < sample_num:
            out = sess.run(
                tf_sample,
                feed_dict={context: batch_size * [context_tokens]})
            for i in range(min(sample_num - index, batch_size)):
                text = enc.decode(out[i])
                text = '======== SAMPLE {} ========\n{}\n'.format(
                    index + 1, text)
                all_text.append(text)
                index += 1
        print(text)
        maketree(os.path.join(SAMPLE_DIR, run_name))
        with open(
                os.path.join(SAMPLE_DIR, run_name,
                             'samples-{}').format(counter), 'w', encoding='utf8') as fp:
            fp.write('\n'.join(all_text))

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(batch_size)]

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))
        save()

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                save()
                return
            if (counter - 1) % save_every == 0 and counter > 1:
                save()
            if (counter - 1) % sample_every == 0 and counter > 1:
                generate_samples()

            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summary_loss),
                    feed_dict={context: sample_batch()})

            summary_log.add_summary(v_summary, counter)

            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                        .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
            current_step += 1
            assign_op = global_step.assign(current_step)
            sess.run(assign_op)
    except KeyboardInterrupt:
        print('interrupted')
        save()


def one_lr_cycle(sess,
                 dataset,
                 steps=10000,
                 model_name='117M',
                 combine=50000,
                 batch_size=1,
                 intial_lr=1e-10,
                 final_lr=1,
                 accumulate_gradients=5,
                 restore_from='fresh',
                 run_name='run1',
                 max_checkpoints=1,
                 use_memory_saving_gradients=False,
                 only_train_transformer_layers=False,
                 overwrite=False):
    """Does one LR half-cycle from initial to final over steps iterations using CLR algorithm
    https://github.com/bckenstler/CLR

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    CHECKPOINT_DIR = 'checkpoint'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        if file not in files:
            try:
                shutil.copyfile(os.path.join('models', model_name, file),
                                os.path.join(checkpoint_path, file))
            except FileNotFoundError as fnf_error:
                print("You need to download the GPT-2 model first via download_gpt2()")
                raise (fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if model_name != '117M':
        use_memory_saving_gradients = True
        only_train_transformer_layers = True
        accumulate_gradients = 1

    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    current_iter = 0
    learning_rate = tf.placeholder(tf.float32, shape=[])

    def get_lr():
        cycle = np.floor(1 + current_iter / (2 * steps))
        x = np.abs(current_iter / steps - 2 * cycle + 1)
        lr = intial_lr + (final_lr - intial_lr) * np.maximum(0, (1 - x))  # * scale_fn(x)
        return lr

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars
    if accumulate_gradients > 1:
        if use_memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=tf.train.AdamOptimizer(learning_rate=learning_rate),
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if use_memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_log = tf.summary.FileWriter(checkpoint_path)

    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join('models', model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(batch_size)]

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))

    start_time = time.time()

    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                return
            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch(), learning_rate: get_lr()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summary_loss),
                    feed_dict={context: sample_batch(), learning_rate: get_lr()})

            summary_log.add_summary(v_summary, counter)

            print('[{counter} | {time:2.2f}] loss={loss:3.14f} lr={lr:2.14f}'
                .format(
                counter=counter,
                time=time.time() - start_time,
                loss=v_loss,
                lr=get_lr()))

            counter += 1
            current_iter += 1
    except KeyboardInterrupt:
        print('interrupted')


def load_gpt2(sess,
              run_name="run1"):
    """Loads the model checkpoint into a TensorFlow session
    for repeated predictions.
    """

    CHECKPOINT_DIR = 'checkpoint'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.placeholder(tf.int32, [1, None])
    output = model.model(hparams=hparams, X=context)

    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver = tf.train.Saver(allow_empty=True)
    sess.run(tf.global_variables_initializer())

    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)


def generate(sess,
             run_name='run1',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             top_p=0.0,
             include_prefix=True):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix:
        context = tf.placeholder(tf.int32, [batch_size, None])

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    np.random.seed(seed)
    tf.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )[:, 1:]

    if destination_path:
        f = open(destination_path, 'w')
    if prefix:
        context_tokens = enc.encode(prefix)
    generated = 0
    gen_texts = []
    while generated < nsamples:
        if not prefix:
            out = sess.run(output)
        else:
            out = sess.run(output, feed_dict={
                context: batch_size * [context_tokens]
            })
        for i in range(batch_size):
            generated += 1
            gen_text = enc.decode(out[i])
            if prefix:
                gen_text = prefix[0] + gen_text
            if truncate:
                truncate_esc = re.escape(truncate)
                if prefix and not include_prefix:
                    prefix_esc = re.escape(prefix)
                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                         truncate_esc)
                else:
                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                trunc_text = re.search(pattern, gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim))
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts


def generate_to_file(sess,
                     run_name='run1',
                     truncate=None,
                     destination_path='gpt_2_gen_texts.txt',
                     sample_delim='=' * 20 + '\n',
                     prefix=None,
                     seed=None,
                     nsamples=1,
                     batch_size=1,
                     length=1023,
                     temperature=0.7,
                     top_k=0,
                     top_p=0.0,
                     include_prefix=True):
    """Generates the texts to a file.

    sample_delim separates texts: set to '' if each text is a small document.

    Adapted from https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn.py
    """

    generate(sess,
             run_name,
             False,
             truncate,
             destination_path,
             sample_delim,
             prefix,
             seed,
             nsamples,
             batch_size,
             length,
             temperature,
             top_k,
             top_p,
             include_prefix)


def mount_gdrive():
    """Mounts the user's Google Drive in Colaboratory."""
    assert 'google.colab' in sys.modules, "You must be in Colaboratory to mount your Google Drive"

    drive.mount('/content/drive')


def is_mounted():
    """Checks if the Google Drive is mounted."""
    assert os.path.isdir('/content/drive'), "You must mount first using mount_gdrive()"


def get_tarfile_name(checkpoint_folder):
    """Converts a folder path into a filename for a .tar archive"""
    tarfile_name = checkpoint_folder.replace(os.path.sep, '_') + '.tar'

    return tarfile_name


def copy_checkpoint_to_gdrive(run_name='run1', copy_folder=False):
    """Copies the checkpoint folder to a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join('checkpoint', run_name)

    if copy_folder:
        shutil.copytree(checkpoint_folder, "/content/drive/My Drive/" + checkpoint_folder)
    else:
        file_path = get_tarfile_name(checkpoint_folder)

        # Reference: https://stackoverflow.com/a/17081026
        with tarfile.open(file_path, 'w') as tar:
            tar.add(checkpoint_folder)

        shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_checkpoint_from_gdrive(run_name='run1', copy_folder=False):
    """Copies the checkpoint folder from a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join('checkpoint', run_name)

    if copy_folder:
        shutil.copytree("/content/drive/My Drive/" + checkpoint_folder, checkpoint_folder)
    else:
        file_path = get_tarfile_name(checkpoint_folder)

        shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)

        with tarfile.open(file_path, 'r') as tar:
            tar.extractall()


def copy_file_to_gdrive(file_path):
    """Copies a file to a mounted Google Drive."""
    is_mounted()

    shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_file_from_gdrive(file_path):
    """Copies a file from a mounted Google Drive."""
    is_mounted()

    shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)


def is_gpt2_downloaded(model_name='117M'):
    """Checks if the original model + associated files are present in folder."""

    for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                     'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                     'model.ckpt.meta', 'vocab.bpe']:
        if not os.path.isfile(os.path.join("models", model_name, filename)):
            return False
    return True


def encode_csv(csv_path, out_path='csv_encoded.txt', header=True,
               start_token="<|startoftext|>",
               end_token="<|endoftext|>"):
    """Encodes a single-column CSV to a format suitable for gpt-2-simple.
       Automatically adds the specified prefix and suffix tokens.
    """

    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f:
        with open(out_path, 'w', encoding='utf8', errors='ignore') as w:
            if header:
                f.readline()
            reader = csv.reader(f, quoting=csv.QUOTE_ALL)
            for row in reader:
                w.write(start_token + row[0] + end_token + "\n")


def encode_dataset(file_path, out_path='text_encoded.npz',
                   model_name="117M",
                   combine=50000):
    """Preencodes a text document into chunks and compresses it,
    saving time when generated.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/encode.py
    """

    model_path = os.path.join('models', model_name)
    enc = encoder.get_encoder(model_path)
    print('Reading files')
    chunks = load_dataset(enc, file_path, combine)
    print('Writing', out_path)
    np.savez_compressed(out_path, *chunks)


def cmd():
    """Function called when invoking from the terminal."""

    parser = argparse.ArgumentParser(
        description="Easily retrain OpenAI's GPT-2 text-generating model on new texts. (https://github.com/minimaxir/gpt-2-simple)"
    )

    # Explicit arguments

    parser.add_argument(
        '--mode', help='Mode for using the CLI (either "finetune" or "generate") [Required]', nargs='?')
    parser.add_argument(
        '--run_name', help="[finetune/generate] Run number to save/load the model",
        nargs='?', default='run1')
    parser.add_argument(
        '--model_name', help="[finetune] Name of the GPT-2 model to finetune",
        nargs='?', default='117M')
    parser.add_argument(
        '--dataset', help="[finetune] Path to the source text.",
        nargs='?', default=None)
    parser.add_argument(
        '--steps', help="[finetune] Number of steps to train (-1 for infinite)",
        nargs='?', default=-1)
    parser.add_argument(
        '--restore_from', help="[finetune] Whether to load model 'fresh' or from 'latest' checkpoint.",
        nargs='?', default='latest')
    parser.add_argument(
        '--sample_every', help="[finetune] After how many steps to print sample",
        nargs='?', default=1000000, type=int)
    parser.add_argument(
        '--save_every', help="[finetune] After how many steps to save checkpoint",
        nargs='?', default=100, type=int)
    parser.add_argument(
        '--print_every', help="[finetune] After how many steps to print progress",
        nargs='?', default=10, type=int)
    parser.add_argument(
        '--overwrite', help="[finetune] Overwrite existing model when continuing training",
        nargs='?', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--nfiles', help="[generate] How many files to generate.",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--nsamples', help="[generate] How many texts to generate.",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--folder', help="[generate] Folder to save the generated files",
        nargs='?', default="gen", type=str)
    parser.add_argument(
        '--length', help="[generate] Length (tokens) of the generated texts",
        nargs='?', default=1023, type=int)
    parser.add_argument(
        '--temperature', help="[generate] Temperature of the generated texts",
        nargs='?', default=0.7, type=float)
    parser.add_argument(
        '--top_k', help="[generate] Sample only from top k tokens",
        nargs='?', default=0, type=int)
    parser.add_argument(
        '--top_p', help="[generate] Sample from top p prob (overrides top_k if nonzero)",
        nargs='?', default=0.0, type=float)
    parser.add_argument(
        '--batch_size', help="[generate] Batch size for generation (increase for GPUs)",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--prefix', help="[generate] Prefix for generated texts",
        nargs='?', default=None)
    parser.add_argument(
        '--truncate', help="[generate] Truncation for generated texts",
        nargs='?', default=None)
    # https://stackoverflow.com/a/46951029
    parser.add_argument(
        '--include_prefix', help="[generate] Include prefix when truncating.",
        nargs='?', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--sample_delim', help="[generate] Delimiter between each generated sample.",
        nargs='?', default='=' * 20 + '\n', type=str)

    # Positional arguments
    parser.add_argument('mode', nargs='?')
    parser.add_argument('dataset', nargs='?')

    args = parser.parse_args()
    assert args.mode in ['finetune', 'generate'], "Mode must be 'finetune' or 'generate'"

    if args.mode == 'finetune':
        assert args.dataset is not None, "You need to provide a dataset."

        cmd_finetune(dataset=args.dataset, run_name=args.run_name,
                     model_name=args.model_name,
                     steps=args.steps, restore_from=args.restore_from,
                     sample_every=args.sample_every,
                     save_every=args.save_every,
                     print_every=args.print_every,
                     overwrite=args.overwrite)
    if args.mode == "generate":
        cmd_generate(nfiles=args.nfiles, nsamples=args.nsamples,
                     folder=args.folder, length=args.length,
                     temperature=args.temperature, batch_size=args.batch_size,
                     prefix=args.prefix, truncate=args.truncate,
                     include_prefix=args.include_prefix,
                     sample_delim=args.sample_delim, run_name=args.run_name,
                     top_k=args.top_k, top_p=args.top_p)


def cmd_finetune(dataset, run_name, model_name, steps,
                 restore_from, sample_every,
                 save_every, print_every, overwrite):
    """Wrapper script for finetuning the model via the CLI."""

    if not is_gpt2_downloaded(model_name=model_name):
        download_gpt2(model_name=model_name)

    sess = start_tf_sess()
    finetune(sess, dataset=dataset, run_name=run_name,
             model_name=model_name,
             steps=steps, restore_from=restore_from,
             sample_every=sample_every, save_every=save_every,
             print_every=print_every,
             overwrite=overwrite)


def cmd_generate(nfiles, nsamples, folder,
                 length, temperature, batch_size,
                 prefix, truncate, include_prefix,
                 sample_delim, run_name,
                 top_k, top_p):
    """Wrapper script for generating text via the CLI.
    The files are generated into a folder, which can be downloaded
    recursively by downloading the entire folder.
    """

    sess = start_tf_sess()
    load_gpt2(sess, run_name=run_name)

    try:
        os.mkdir(folder)
    except:
        shutil.rmtree(folder)
        os.mkdir(folder)

    for _ in trange(nfiles):
        gen_file = os.path.join(folder,
                                'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow()))

        generate_to_file(sess,
                         destination_path=gen_file,
                         length=length,
                         temperature=temperature,
                         nsamples=nsamples,
                         batch_size=batch_size,
                         prefix=prefix,
                         truncate=truncate,
                         include_prefix=include_prefix,
                         sample_delim=sample_delim,
                         top_k=top_k,
                         top_p=top_p
                         )


# Source: https://github.com/mhmoodlan/cyclic-learning-rate

def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):
    """Applies cyclic learning rate (CLR).
       From the paper:
       Smith, Leslie N. "Cyclical learning
       rates for training neural networks." 2017.
       [https://arxiv.org/pdf/1506.01186.pdf]
        This method lets the learning rate cyclically
       vary between reasonable boundary values
       achieving improved classification accuracy and
       often in fewer iterations.
        This code varies the learning rate linearly between the
       minimum (learning_rate) and the maximum (max_lr).
        It returns the cyclic learning rate. It is computed as:
         ```python
         cycle = floor( 1 + global_step /
          ( 2 * step_size ) )
        x = abs( global_step / step_size – 2 * cycle + 1 )
        clr = learning_rate +
          ( max_lr – learning_rate ) * max( 0 , 1 - x )
         ```
        Polices:
          'triangular':
            Default, linearly increasing then linearly decreasing the
            learning rate at each cycle.
           'triangular2':
            The same as the triangular policy except the learning
            rate difference is cut in half at the end of each cycle.
            This means the learning rate difference drops after each cycle.
           'exp_range':
            The learning rate varies between the minimum and maximum
            boundaries and each boundary value declines by an exponential
            factor of: gamma^global_step.
         Example: 'triangular2' mode cyclic learning rate.
          '''python
          ...
          global_step = tf.Variable(0, trainable=False)
          optimizer = tf.train.AdamOptimizer(learning_rate=
            clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
          train_op = optimizer.minimize(loss_op, global_step=global_step)
          ...
           with tf.Session() as sess:
              sess.run(init)
              for step in range(1, num_steps+1):
                assign_op = global_step.assign(step)
                sess.run(assign_op)
          ...
           '''
         Args:
          global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
            Global step to use for the cyclic computation.  Must not be negative.
          learning_rate: A scalar `float32` or `float64` `Tensor` or a
          Python number.  The initial learning rate which is the lower bound
            of the cycle (default = 0.1).
          max_lr:  A scalar. The maximum learning rate boundary.
          step_size: A scalar. The number of iterations in half a cycle.
            The paper suggests step_size = 2-8 x training iterations in epoch.
          gamma: constant in 'exp_range' mode:
            gamma**(global_step)
          mode: one of {triangular, triangular2, exp_range}.
              Default 'triangular'.
              Values correspond to policies detailed above.
          name: String.  Optional name of the operation.  Defaults to
            'CyclicLearningRate'.
         Returns:
          A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
          learning rate.
        Raises:
          ValueError: if `global_step` is not supplied.
        @compatibility(eager)
        When eager execution is enabled, this function returns
        a function which in turn returns the decayed learning
        rate Tensor. This can be useful for changing the learning
        rate value across different invocations of optimizer functions.
        @end_compatibility
    """
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle = math_ops.floor(math_ops.add(1., global_div_double_step))
            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))
            # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)
            if mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                    cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()
        return cyclic_lr
