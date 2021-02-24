# coding=utf-8
# Copyright 2021 The SemSAD Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scipy.io import wavfile
import os 
import h5py
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default='dataset/LibriSpeech/train-clean-100/', type=str)
    parser.add_argument("--data_root", default='dataset', type=str)
    args = parser.parse_args()
    
    trainroot = [args.train_root]

    """convert wav files to raw wave form and store them 
    """
    
    h5f = h5py.File(os.path.join(args.data_root, 'train-Librispeech.h5'), 'w')
    for rootdir in trainroot:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith('.wav'):
                    fullpath = os.path.join(subdir, file)
                    fs, data = wavfile.read(fullpath)
                    h5f.create_dataset(file[:-4], data=data)
                    print(file[:-4])
    h5f.close()
