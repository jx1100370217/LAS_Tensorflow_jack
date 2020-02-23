# coding=utf-8
"""
@File: run.py
-----------------------------------------
@Author: Jack
@Time: 2020/2/22/022 15:41
@Email:jack18588951684@163.com
-----------------------------------------
"""
import numpy as np
import tensorflow as tf
import utils
from config.config import *
from vctk.preprocess import process_speakers
from vctk.preprocess import parse_args as parse_args_pre
from train import parse_args as parse_args
from infer import parse_args as parse_args_infer
from train import input_fn
from infer import input_fn as input_fn_infer
from vctk.write_tfrecord import make_example
from model_helper import las_model_fn


def process_data():
    args = parse_args_pre()
    speakers = list(sorted(os.listdir(args.txt_dir)))
    train_speakers, test_speakers = speakers[:4], speakers[4:]
    print('Process training')
    features, labels = process_speakers(train_speakers, args)
    np.save(TRAIN_FEATURE, features)
    np.save(TRAIN_LABEL, labels)

    print('Process testing')
    features, labels = process_speakers(test_speakers, args)
    np.save(TEST_FEATURE, features)
    np.save(TEST_LABEL, labels)


def build_vocab(label_dir, vocab_dir):
    s = set()
    f = np.load(label_dir, allow_pickle=True).item()
    for line in f.values():
        s.update(line)
    d = sorted(list(s))
    with open(vocab_dir, 'w') as f:
        print('\n'.join(d), file=f)


def npy2tfrecord(inputs_dir, labels_dir, output_dir):
    with tf.io.TFRecordWriter(output_dir) as writer:
        inputs = np.load(inputs_dir, allow_pickle=True).item()
        labels = np.load(labels_dir, allow_pickle=True).item()
        assert len(inputs) == len(labels)
        for i, (name, input) in enumerate(inputs.items()):
            label = labels[name]
            if i < 10:
                print(name, label)
            writer.write(make_example(input, label).SerializeToString())


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    args.train = TRAIN_TF
    args.valid = TEST_TF
    args.vocab = VOCAB_TABLE
    args.model_dir = MODEL_DIR
    vocab_list = utils.load_vocab(args.vocab)
    vocab_size = len(vocab_list)

    conf = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)

    model = tf.estimator.Estimator(
        model_fn=las_model_fn,
        config=conf,
        params=hparams)

    if args.valid:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(
                args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size,
                num_epochs=args.num_epochs))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(
                args.valid or args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size),
            start_delay_secs=60, throttle_secs=args.eval_secs)

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        model.train(
            input_fn=lambda: input_fn(
                args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size,
                num_epochs=args.num_epochs))


def infer():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args_infer()
    args.model_dir = MODEL_DIR
    args.beam_width = 32
    args.data = TEST_TF
    args.save = INFER_RESULT
    args.vocab = VOCAB_TABLE
    vocab_list = np.array(utils.load_vocab(args.vocab))
    vocab_size = len(vocab_list)

    conf = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)
    hparams.decoder.set_hparam('beam_width', args.beam_width)

    model = tf.estimator.Estimator(
        model_fn=las_model_fn, config=conf, params=hparams)
    predictions = model.predict(
        input_fn=lambda: input_fn_infer(
            args.data, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size, num_epochs=1,),
        predict_keys='sample_ids')
    if args.beam_width > 0:
        predictions = [vocab_list[y['sample_ids'][:, 0]].tolist() + [utils.EOS] for y in predictions]
    else:
        predictions = [vocab_list[y['sample_ids']].tolist() + [utils.EOS] for y in predictions]

    predictions = [' '.join(y[:y.index(utils.EOS)]) for y in predictions]
    with open(args.save, 'w') as f:
        f.write('\n'.join(predictions))


if __name__ == '__main__':
    ## step1:数据预处理
    # process_data()
    ## step2:生成TF_Record数据(train.tfrecord,test.tfrecord)
    # npy2tfrecord(TRAIN_FEATURE, TRAIN_LABEL, TRAIN_TF)
    # npy2tfrecord(TEST_FEATURE, TEST_LABEL, TEST_TF)
    ## step3:构建词表
    # build_vocab(TRAIN_LABEL, VOCAB_TABLE)
    ## step4:模型训练
    # train()
    ## step5:模型评估
    infer()
