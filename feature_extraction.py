from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("hello.wav")
print(rate)
print(sig)
mfcc_feat = mfcc(sig,rate)
print(mfcc_feat)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])

import numpy as np
import pandas as pd
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import pickle
filename='model.pkl'
classifier=pickle.load(open(filename, 'rb'))
audioData = []
srate = []

data,sr = lb.load("hello.wav")
mfcc = librosa.feature.mfcc(y=data, sr=sr)
print(mfcc)

mfccScaled = np.mean(mfcc.T, axis=0)
print(mfccScaled)
print(mfccScaled.shape)
plt.plot(mfccScaled, 'g')
plt.show()