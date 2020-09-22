import numpy as np
import tensorflow as tf
import keras
import glob
import math
import random
from random import getrandbits

from matplotlib import pyplot as plt
from keras.preprocessing import image


def getFiles(stack_dir, mask_dir):
    '''
    ** find all stack images in dir., along with matching mask image files **
    ~~~~~~~~~
    INPUTS:
            - stack_dir = directory containing all stack images
            - mask_dir = directory containing all mask images
    RETURNS:
            - 2d array of stack and mask pairs [[stack_image_loc, mask_image_loc], ...]
    '''
    stacks = []
    masks = []
    for stack_image in glob.glob(stack_dir+"/*.npy"):
        x, y, z = stack_image.split(
            stack_dir+"/stack_tile_")[1].split(".npy")[0].split("_")
        mask_image = mask_dir+"/mask_tile_{}_{}_{}.npy".format(x, y, z)
        stacks.append(stack_image)
        masks.append(mask_image)
    return stacks, masks


class dataset_gen(keras.utils.Sequence):
    ' See https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence'

    def __init__(self, stacks, masks, batch_size):
        self.stacks = stacks
        self.masks = masks
        self.batch_size = batch_size

    def __len__(self):
        ' number of batches per epoch '
        return math.floor(len(self.stacks) / self.batch_size)

    def __getitem__(self, index):
        # print("\n\n\n{}\n\n\n".format(index))
        indices = [index*self.batch_size, (index+1)*self.batch_size]
        #print("Index: {}\nIndices: {}\nDifferences: {}".format(index,indices, (indices[1]-indices[0])))
        i = indices[0]
        temp = []
        while i < indices[1]:
            temp.append(i)
            i = i+1

        X, y = self.__data_generation(temp)

        return X, y

    def __data_generation(self, list_IDs_temp):

        augment=True # augment date (y/N)

        X = np.empty(
            (self.batch_size, *(256, 256), 5))
        Y = np.empty(
            (self.batch_size, *(256, 256), 6))

        for i, ID in enumerate(list_IDs_temp):
            # random values to determine augmentation
            flip_horo=bool(getrandbits(1))
            flip_vert=bool(getrandbits(1))
            adjust_brightness=bool(getrandbits(1))
            zoom=bool(getrandbits(1))

            brightness_delta=random.randint(0,6)/10
            movement=random.randint(0,3)/10
            #print("\n\n\n{}\n\n".format(self.masks[ID]))
            if augment:
                X[i, ] = self.__augment_np(np.load(self.stacks[ID]), flip_horo, flip_vert, adjust_brightness, zoom, brightness_delta, movement) # apply augmentation here
                Y[i, ] = self.__augment_np(np.load(self.masks[ID]), flip_horo, flip_vert, adjust_brightness, zoom, brightness_delta, movement) # apply augmentation here
            else:
                X[i, ] = np.load(self.stacks[ID]) # apply augmentation here
                Y[i, ] = np.load(self.masks[ID]) # apply augmentation here

        return X, Y


    def __augment_np(self, np_image, flip_horo, flip_vert, adjust_brightness, zoom, brightness_delta, movement):
        ' WORK IN PROGRESS... (NEED TO AUTOMATE LR CHANGE TO TAKE ADVANTAGE, REALLY'
        if flip_horo:
            #np_image=tf.image.flip_left_right(np_image)
            np_image=np.flip(np_image, 1)

        if flip_vert:
            #np_image=tf.image.flip_up_down(np_image)
            np_image=np.flip(np_image, 0)

        '''
        # disabled for debugging...
        if adjust_brightness:
            brightness_delta=brightness_delta
            np_image=tf.image.adjust_brightness(np_image, brightness_delta)

        if zoom:
            width=height=256
            offset_h=math.floor(movement*height)
            offset_w=math.floor(movement*width)
            movement=movement
            np_image=tf.image.crop_to_bounding_box(image=np_image,offset_height=offset_h, offset_width=offset_w, target_height=(height-offset_h), target_width=(width-offset_w))
            np_image=tf.image.resize(images=np_image, size=[256,256])
        '''
        return np_image
