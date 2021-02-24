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