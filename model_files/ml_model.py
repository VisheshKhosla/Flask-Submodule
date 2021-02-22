# import the libraries

import os
from flask import Flask,jsonify
import json
import sys
import librosa 
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


# audio_file='C:/Users/Dell/Desktop/Flask_App/model_files/audio.mp3'

# Some Important constants

DUR=60
NUM_MFCC=13
NUM_FFT=2048
HOP_LENGTH=512
SAMPLE_RATE=22050
SAMPLES_PER_TRACK=22050*DUR
NUM_SEGMENTS=20
SAMPLES_PER_SEGMENT=(SAMPLES_PER_TRACK)/(NUM_SEGMENTS)
mfcc_vectors_per_segment=math.ceil((SAMPLES_PER_SEGMENT)/(HOP_LENGTH))

def predict_music(model,audio_file):

	# Loading the audio file
	signal,sr=librosa.load(audio_file,sr=SAMPLE_RATE)

	pred=[]
	for curr in range(NUM_SEGMENTS):

        # Creating MFCC vectors of a segment

		start_sample=SAMPLES_PER_SEGMENT*curr 
		finish_sample=start_sample+SAMPLES_PER_SEGMENT
		start_sample=int(start_sample)
		finish_sample=int(finish_sample)
		mfcc=librosa.feature.mfcc(signal[start_sample:finish_sample],
			sr=SAMPLE_RATE,n_fft=NUM_FFT,n_mfcc=NUM_MFCC
			,hop_length=HOP_LENGTH)
		mfcc=mfcc.T

		# Reshaping the array for a valid i/p
		
		np_mfcc=np.array(mfcc)
		np_mfcc=np_mfcc.reshape(1,mfcc_vectors_per_segment,NUM_MFCC,1)
		mfcc=np_mfcc

		pred.append(model.predict(mfcc))

	final_pred=np.zeros(10)

	# Analysing all of the segments(one hot encoded form)

	for curr in range(len(pred)):
	    predicted_index=0
	    maxpred=0
	    for genre in range(10):
	        if(pred[curr][0][genre]>maxpred):
	            maxpred=pred[curr][0][genre]
	            predicted_index=genre
	    final_pred[predicted_index]+=1

    # Calculating percentage
	for genre in range(10):
		final_pred[genre]=(final_pred[genre]/20)*100

	# NP Array-> List
	return final_pred.tolist()
	 	
	
