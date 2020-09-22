import sys
from time import localtime, strftime
import json
import random

import tensorflow as tf
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.losses import binary_crossentropy

from model_def import *
from data_generator import *
from callbacks import *

# inputs
stacked_image_dir='/home/jordan/Documents/Github/resunet-tf/Data/npy/stack'
mask_dir='/home/jordan/Documents/Github/resunet-tf/Data/tensors'

# get files from stack and mask dir.
stacks, masks = getFiles(stacked_image_dir, mask_dir)

# shuffle the datasets so that all features is represented in the training dataset
seed=4
random.Random(seed).shuffle(stacks)
random.Random(seed).shuffle(masks)

stacks_len=len(stacks)

# hyperparams.
TRAIN_LENGTH = int(stacks_len*0.9)
EPOCHS = 125
BATCH_SIZE = 2
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#STEPS_PER_EPOCH=5 # used for dev.
VAL_SUBSPLITS=5
VALIDATION_STEPS=int(stacks_len*0.10)//BATCH_SIZE//VAL_SUBSPLITS

# get train and test datasets from generator (dataset_gen)
train_dataset = dataset_gen(stacks[:TRAIN_LENGTH], masks[:TRAIN_LENGTH], BATCH_SIZE)
test_dataset = dataset_gen(stacks[1+TRAIN_LENGTH:], masks[1+TRAIN_LENGTH:], BATCH_SIZE)

# define model (img. width, img. height, in. channels, and out. channels)
model = resunet_a_d6(256, 256, 5, 6)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),loss=focal_tversky_loss, metrics=['accuracy', tversky, tf.keras.metrics.MeanIoU(num_classes=6)])

model.summary()

# Loads historic weights weights
'''
checkpoint_path='/run/user/1000/gvfs/smb-share:server=192.168.1.30,share=jordan/Dump/output_11/model-51-0.47.hdf5'
model.load_weights(checkpoint_path)
'''

model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_dataset, batch_size=BATCH_SIZE, verbose=1, callbacks=[LogTraining(), save_checkpoints])

# save the model
model.save('model_ckpt/')