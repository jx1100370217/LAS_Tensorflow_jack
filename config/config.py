# coding=utf-8
"""
@File: config.py
-----------------------------------------
@Author: Jack
@Time: 2020/2/22/022 15:36
@Email:jack18588951684@163.com
-----------------------------------------
"""
import os

DATA_PATH = os.path.join(os.getcwd(),'data')

VCTK_DIR = os.path.join(DATA_PATH,'VCTK-Corpus')
PROCESSED_DIR = os.path.join(VCTK_DIR,'processed')
TF_DIR = os.path.join(VCTK_DIR,'tfdata')
MODEL_DIR = os.path.join(VCTK_DIR,'model')
INFER_RESULT = os.path.join(VCTK_DIR,'infer_result.txt')

TRAIN_FEATURE = os.path.join(PROCESSED_DIR,'train_feat.npy')
TRAIN_LABEL = os.path.join(PROCESSED_DIR,'train_label.npy')
TEST_FEATURE = os.path.join(PROCESSED_DIR,'test_feat.npy')
TEST_LABEL = os.path.join(PROCESSED_DIR,'test_label.npy')
TRAIN_TF = os.path.join(TF_DIR,'train.tfrecord')
TEST_TF = os.path.join(TF_DIR,'test.tfrecord')
VOCAB_TABLE = os.path.join(TF_DIR,'vocab.table')
