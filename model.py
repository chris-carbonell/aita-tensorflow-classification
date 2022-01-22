# Chris Carbonell
# CSC510
# Module 3

# Dependencies

# general
import csv
import datetime
import os
import re
import shutil
import string
import time

# data
import pandas as pd
import pyarrow.parquet as pq

# tf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

# Constants

# train, validate, test
str_path_train = "./data/train"
str_path_validate = "./data/validate"
str_path_test = "./data/test"

# checkpoints
str_path_checkpoints_root = "./checkpoints"
str_filename_format_checkpoints_model = str_path_checkpoints_root + "/model_cp_{epoch:04d}.ckpt"

# save
str_path_model_save = "./model"

# Funcs

def create_model():
	'''
	train and save the model

	predicted probabilty is probabilty of yes (1)
	'''

	# make checkpoints dir if necessary
	os.makedirs(str_path_checkpoints_root, exist_ok=True)

	# Create Datasets

	batch_size = 32
	seed = 42

	# train
	ds_train = tf.keras.utils.text_dataset_from_directory(
	    str_path_train,
	    batch_size=batch_size,
	    seed=seed)

	# validate
	ds_validate = tf.keras.utils.text_dataset_from_directory(
	    str_path_validate,
	    batch_size=batch_size,
	    seed=seed)

	# test
	ds_test = tf.keras.utils.text_dataset_from_directory(
	    str_path_test,
	    batch_size=batch_size,
	    seed=seed)

	# Standardize, Tokenize, and Vectorize

	## Prep Layer

	max_features = 10000
	sequence_length = 250

	vectorize_layer = layers.TextVectorization(
	    max_tokens=max_features,
	    output_mode='int',
	    output_sequence_length=sequence_length)

	train_text = ds_train.map(lambda text, labels: text)
	vectorize_layer.adapt(train_text)

	## Apply

	ds_train_v = ds_train.map(vectorize_text)
	ds_validate_v = ds_validate.map(vectorize_text)
	ds_test_v = ds_test.map(vectorize_text)

	## Create Model

	embedding_dim = 16

	model = tf.keras.Sequential([
	  layers.Embedding(max_features + 1, embedding_dim),
	  layers.Dropout(0.2),
	  layers.GlobalAveragePooling1D(),
	  layers.Dropout(0.2),
	  layers.Dense(1)])

	# compile
	model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
	              optimizer='adam',
	              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

	# create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
	    filepath=str_filename_format_checkpoints_model, 
	    verbose=1, 
	    save_weights_only=True,
	    save_freq=5*batch_size
	)

	# fit
	epochs = 10
	history = model.fit(
	    ds_train_v,
	    validation_data=ds_validate_v,
	    epochs=epochs,
	    callbacks=[cp_callback]
	)

	# create export model

	export_model = tf.keras.Sequential([
	  vectorize_layer,
	  model,
	  layers.Activation('sigmoid')
	])

	export_model.compile(
	    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
	)

	# save
	export_model.save(str_path_model_save)

	return

if __name__ == "__main__":

	# load model
	model = tf.keras.models.load_model(str_path_model_save)
	os.system('cls')

	# get user input

	while True:

		str_break = "exit"

		# get user input
		test_selftext = input(f'testing text ("{str_break}" to exit): ')

		# break?
		if test_selftext == str_break:
			break
		
		# predit (yes = 1)
		predicted_prob = round(model.predict([test_selftext])[0][0], 3)

		# output results
		print("Prediction (yes = 1):")
		print(predicted_prob)
		print()