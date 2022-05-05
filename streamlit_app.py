#!/usr/bin/env python
# coding: utf-8

import time
import librosa
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
# upload model
#model_all_emo = load_model("lstm_model_all_emo_accuracy_0.6131_val_accuracy_0.5550.h5")
model_4_emo = load_model("lstm_model_4_emo_accuracy_0.9139_val_accuracy_0.8650.h5")

# 提取 mfcc 參數
def wav2mfcc(path, max_pad_size=11):     
    y, sr = librosa.load(path=path, sr=None, mono=False)     
    y = y[::3]
    audio_mac = librosa.feature.mfcc(y=y, sr=16000)     
    y_shape = audio_mac.shape[1]     
    if y_shape < max_pad_size:         
        pad_size = max_pad_size - y_shape         
        audio_mac = np.pad(audio_mac, ((0, 0), (0, pad_size)), mode='constant')     
    else:         
        audio_mac = audio_mac[:, :max_pad_size]     
    return audio_mac

# web page
st.title('情感分析系統')

option = st.selectbox('Please Select', ['sample 1.wav', 'sample 2.wav', 'sample 3.wav', 'sample 4.wav'])

if st.button('Submit'):
    emo_dict = {'angry':0, 'happy':1, 'sad':2, 'calm':3}
    test_mfcc_vector = []
    if option == 'sample 1.wav':
        path = "./voice/03-02-03-01-02-01-07_happy.wav"
    if option == 'sample 2.wav':    
        path = "./voice/03-02-04-02-01-01-07_sad.wav"
    if option == 'sample 3.wav':
        path = "./voice/03-02-05-01-02-01-07_angry.wav"
    if option == 'sample 4.wav':
        path = "./voice/16_01_02_01_kids-talking_fear.wav"

    mfcc = wav2mfcc(path, max_pad_size=50)
    test_mfcc_vector.append(mfcc)
    X_test = np.array(test_mfcc_vector).reshape(-1, 20, 50, 1)
    # 呼叫模型、預測，最後把結果放入dataframe呈現
    pred = model_4_emo.predict(X_test)
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 22})
    emo = ['angry', 'happy', 'sad', 'calm']
    percent = list(pred[0]*100)
    x = np.arange(len(emo))
    plt.figure(figsize=(15,5))
    plt.barh(x, percent, height=0.3)
    plt.yticks(x, emo)
    plt.ylabel('emotion category')
    plt.xlabel('percentage(%)')
    st.pyplot()
