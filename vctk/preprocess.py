import os
import sys
import string
from argparse import ArgumentParser
import numpy as np
import scipy.io.wavfile as wav
import speechpy
from config.config import *

def parse_args():
    parser = ArgumentParser('Process VCTK dataset')

    parser.add_argument('--data_dir', help='root directory of VCTK dataset')
    parser.add_argument(
        '--output_dir', help='output directory of processed dataset')

    args = parser.parse_args()
    args.data_dir = VCTK_DIR
    args.output_dir = PROCESSED_DIR
    args.txt_dir = os.path.join(args.data_dir, 'txt')
    args.wav_dir = os.path.join(args.data_dir, 'wav48')

    return args


def extract_mfcc(filename):
    if not filename.endswith('.wav'):
        return None, None

    fs, signal = wav.read(filename)
    assert fs == 48000

    # downsample
    signal = signal[::3]
    fs = 16000

    mfcc = speechpy.feature.mfcc(signal, fs)
    mfcc_cmvn = speechpy.processing.cmvn(mfcc, True)

    mfcc_39 = speechpy.feature.extract_derivative_feature(mfcc_cmvn)

    return filename[:-4], mfcc_39.reshape(-1, 39)


def parse_trans(filename):
    if not filename.endswith('.txt'):
        return None, None

    with open(filename) as f:
        s = f.readlines()[0].strip()
        s = ' '.join(s.split())
        translator = str.maketrans('', '', string.punctuation)
        s = s.translate(translator).lower()

    return filename[:-4], s.split()


def process_speakers(speakers, args):
    features = {}
    labels = {}
    for speaker in speakers:
        txt_speaker_dir = os.path.join(args.txt_dir, speaker)
        wav_speaker_dir = os.path.join(args.wav_dir, speaker)
        for txt_file, wav_file in zip(sorted(os.listdir(txt_speaker_dir)), sorted(os.listdir(wav_speaker_dir))):
            assert txt_file[:-4] == wav_file[:-4]
            wav_filepath = os.path.join(wav_speaker_dir, wav_file)
            txt_filepath = os.path.join(txt_speaker_dir, txt_file)

            name, x = extract_mfcc(wav_filepath)
            _name, y = parse_trans(txt_filepath)

            if name is not None and _name is not None:
                features[name] = x
                labels[name] = y

    return features, labels
