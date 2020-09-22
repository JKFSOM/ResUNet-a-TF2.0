import json # need this?
import os
import mysql.connector
from time import localtime, strftime # need this?
import tensorflow as tf
from tensorflow import keras
from statistics import *

from mysql_pw import mysql_pw

base_dir='/run/user/1000/gvfs/smb-share:server=192.168.1.30,share=jordan/Dump'
# CHECKPOINT OUTPUT DIR


class LogTraining(keras.callbacks.Callback):
	# LOG TRAINING TO MYSQL DB - OPTIONAL
	def __init__(self):
		self.losses=[]
		self.lrs=[]
		self.instance=21 # used to differentiate training runs (logged in DB and checkpoint names)
		self.last_changed=0 # used for current alter_learning_rate()


	def __connect_to_mysql(self, database):
		# create a mysql connection
		mydb=mysql.connector.connect(
			host="192.168.1.38",
			user="grafana",
			password=mysql_pw,
			database=database
		)
		return mydb

	
	def __insert_records(self, epoch, logs, training_id):
		# insert training outcome of each epoch into MySQL
		mydb=self.__connect_to_mysql('model_progress')
		mycursor=mydb.cursor()

		# get current learning rate
		learningRate=keras.backend.eval(self.model.optimizer.lr).item()

		# store relevant values in database
		sql="INSERT INTO resunet_a_default (trainingID, trainingLoss, trainingAcc, validationLoss, validationAcc, meanIoU, valMeanIoU, learningRate, epoch) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
		vals=(training_id, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy'], logs['mean_io_u'], logs['val_mean_io_u'], learningRate, epoch)

		mycursor.execute(sql, vals)
		mydb.commit()


	def __store_learning_rate(self, epoch, logs):
		# get current learning rate and store in list (num. elements = num. epochs)
		lr=keras.backend.get_value(self.model.optimizer.lr).item()
		self.lrs.append(lr)
		# get loss and store in lost (num. elements = num. epochs)
		val_loss=round(logs['val_loss'], 4)
		self.losses.append(val_loss)
	

	def __mean(self, values):
		# find the mean of values in a list
		total=0.0
		for item in values:
			total=total+item

		return (total/len(values))

	def on_epoch_end(self, epoch, logs=None):
		# at the end of each epoch...

		# Insert the training result for the epoch into the logging DB
		self.__insert_records(epoch=epoch, logs=logs, training_id=self.instance)
		# Log learning rate
		self.__store_learning_rate(epoch, logs)
		# Check if we need to reduce the learning rate
		#self.__alter_learning_rate(epoch)

	
	'''
	** UNUSED (faulty)**
	---
	# CONSIDER USING TRIANGLE ADJUSTMENT - https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
	# concerned that if the change in loss function is not good enough, then it may drop twice in a row
	def __alter_learning_rate(self, epoch):
		#if self.losses[-1] > self.losses[-2] > self.losses[-3]:
		#	# decrease the learning rate
		if len(self.losses) > 7:
			# look at using mean, instead
			if (self.__mean(self.losses[-4:-1]) >= (self.__mean(self.losses[-7:-5]) - 0.25)) and (self.last_changed not in range((epoch-6), epoch)):
				print("\n\nCAUGHT IN BLOCK...")
			#if (round(self.losses[-1], 2) >= round(self.losses[-2], 2) >= round(self.losses[-3], 2) >= round(self.losses[-4], 2)):
				learning_rate=self.lrs[-1]
				if (learning_rate >= 0.00001):
					# model has stopped learning - change the learning rate
					new_learning_rate=learning_rate*0.1
					self.model.optimizer.lr.assign(new_learning_rate)
					self.last_changed=epoch
				elif (learning_rate < 0.00001):
					print("\n\n\nIN IMPORTANT PART OF METHOD...")
					# increase learning rate a bit
					# model has stopped learning - change the learning rate
					print("Less than threshold")
					new_learning_rate=learning_rate*1000
					self.model.optimizer.lr.assign(new_learning_rate)
					print("Assigned new learning rate")
					self.last_changed=epoch
					print("Updated last changed epoch")
					# model has increased 
	'''


def check_dir_existence(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

#instance=21
instance=LogTraining.instance

# SAVE CHECKPOINT WEIGHTS
# ---
# save all weights
# (source https://lambdalabs.com/blog/tensorflow-2-0-tutorial-03-saving-checkpoints/)
output_folder = base_dir+'/output_'+str(instance)
# check directory exists, or create
check_dir_existence(output_folder)

filepath=output_folder+"/model-{epoch:02d}-{val_mean_io_u:.2f}.hdf5"

save_checkpoints=tf.keras.callbacks.ModelCheckpoint(
	filepath=filepath,
	save_weights_only=True,
	save_freq='epoch',
	verbose=1
)

# save the best weights, too - doubling up on checkpoint weights, but saves time for dev.
#checkpoint_path='/best_train_ckpt/cp.ckpt'
checkpoint_path=output_folder+'/best_train_cp.ckpt'
save_best_checkpoints=tf.keras.callbacks.ModelCheckpoint(
	filepath=os.path.dirname(checkpoint_path),
	save_weights_only=True,
	monitor='val_mean_io_u',
	mode='max',
	save_best_only=True,
	save_freq='epoch'
)


'''
** UNUSED (faulty) **
---
# stop training after x epochs if validation loss doesn't decrease by at least 1%
early_stop=tf.keras.callbacks.EarlyStopping(
	monitor='val_mean_io_u',
	min_delta=1,
	patience=25,
	verbose=1,
	mode='max'
)
'''