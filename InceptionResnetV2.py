#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:36:38 2019

@author: abhishekverma
"""

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import time

image = load_img('example.jpg', target_size=(299, 299))


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = InceptionResNetV2()
    # re-structure the model
    
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    #model = load_model('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    
    # load the photo
    image = load_img(filename, target_size=(299, 299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

t1=time.time()
# load the tokenizer
tokenizer = load(open('ImgCap_V3_ResNet/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
#download = drive.CreateFile({'id': '10mMJv6xoKqIMPv4Q1ps8tgbha9_IDWnb'})
#download.GetContentFile('model_4_3.h5')

InceptionResnetV2 = load_model('ImgCap_V3_ResNet/model-ep003-loss3.433-val_loss3.683.h5')
# load and prepare the photograph
#fname='Flickr8k/Flicker8k_Dataset/145721498_a27d2db576.jpg'

fname='example.jpg'

photo = extract_features(fname)
# generate description
description = generate_desc(InceptionResnetV2, tokenizer, photo, max_length)
#print(description)
t2=time.time()
print('time taken: ',t2-t1)
output_str=' '.join(description.split()[1:-1])
print(output_str)
