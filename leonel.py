#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:08:36 2018

@author: leonel
"""

import math
from keras.models import load_model
import numpy as np
import contextlib
import wave
import struct
import sox




def frecuencia(archivo):
    fname = archivo
    with contextlib.closing(wave.open(fname,'r')) as f:
        frate = f.getframerate()
        data_size = f.getnframes()
    wav_file = wave.open(fname, 'r')
    data = wav_file.readframes(data_size)
    data_size = data_size * wav_file.getnchannels()
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=data_size), data)
    data = np.array(data)

    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * frate)
    return int(freq_in_hertz)


def prediction(gender, age, dsmt, hare, ciep, cief, ciec, ciem,cie):
    audio = "output.wav"
    if gender == 'Masculino':
        sex = 0
    elif gender == 'Femenino':
        sex = 1
    transform = sox.Transformer()
    stats = transform.stats(audio)
    rmsTr = stats['RMS Tr dB']
    pkLevel = stats['Pk lev dB']
    crest = stats['Crest factor']
    rmsPk = stats['RMS Pk dB']
    rmsLevel = stats['RMS lev dB']
    length = stats['Length s']
    freq = frecuencia(audio)
    model = load_model('model_Leonel.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data = np.array([[sex, age, dsmt, hare, ciep, cief, ciec, ciem,cie,	100,rmsTr,  pkLevel,crest,rmsPk,rmsLevel,freq, round(float(length))]])
    classes = model.predict_classes(data)
    proba = model.predict(data)
    if classes == [[0]]:
        response = "false"
    elif classes == [[1]]:
        response = "true"
    per = proba[0][0]
    trust = round(per * 100,2)
    return response, trust

