import numpy as np
import pandas as pd
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
audioData = []
srate = []

data,sr = lb.load("hello.wav")
mfcc = librosa.feature.mfcc(y=data, sr=sr)
print(mfcc)

mfccScaled = np.mean(mfcc.T, axis=0)
print(mfccScaled)
X=np.array(mfccScaled)

y_pred =classifier.predict(X)

print(y_pred)