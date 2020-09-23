from tensorflow.keras.layers import Input
import tensorflow as tf
import math
import numpy as np
import sys
from tensorflow import reduce_mean, reduce_sum, where, fill, shape, ones_like, multiply


smooth=1.0e-5
axis=[2,3] # was 2,3

class LossFuncs:
    ' Using Tversky until fix Tanimoto loss func.'
    # Focal Tversky loss - try changing smooth to 1e-5 and __ing functions
    # (taken from https://github.com/nabsabraham/focal-tversky-unet and https://www.kaggle.com/ekhtiar/resunet-a-baseline-on-tensorflow#Loss-Functions)
    def tversky(self, y_true, y_pred, smooth=1e-6):
        y_true_pos = tf.keras.layers.Flatten()(y_true)
        y_pred_pos = tf.keras.layers.Flatten()(y_pred)
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
        false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    # unused...
    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky(y_true,y_pred)

    def focal_tversky_loss(self, y_true, y_pred):
        pt_1 = self.tversky(y_true, y_pred)
        gamma = 0.75
        return tf.keras.backend.pow((1-pt_1), gamma)

    '''
    **
    FAULTY - LOOKING INTO...
    **
    ---
    def tan_loss(y_actual, y_pred):
        # find the mean volume of the class
        Vli=reduce_mean(reduce_sum(y_actual, axis=axis),axis=0)
        # "weighting scheme"
        wli=tf.math.reciprocal(Vli**2)
        # turn inf. elements to zero
        new_weights=where(tf.math.is_inf(wli), fill(shape(wli), 0.0), wli)
        # ...replace zeros with max value <- missing broadcast here...
        wli=where(tf.math.is_inf(wli), multiply(ones_like(wli), tf.keras.backend.max(new_weights)), wli)

        rl_x_pl = reduce_sum(multiply(y_actual, y_pred), axis=axis)
        # sum of squares
        l=reduce_sum(multiply(y_actual, y_actual), axis=axis)
        r=reduce_sum(multiply(y_pred, y_pred), axis=axis)

        rl_p_pl = l + r - rl_x_pl

        numerator=reduce_sum(multiply(wli, rl_x_pl), axis=1)+smooth # was 1
        denominator=reduce_sum(multiply(wli, (rl_p_pl)), axis=1) + smooth # was 1
        tnmt=numerator/denominator

        return tnmt

    def tan_dual(y_actual, y_pred):
        # measure of overlap
        loss1=tan_loss(y_pred, y_actual)

        # measure of non-overlap as inner product
        preds_dual=1.0-y_pred
        actual_dual=1.0-y_actual
        loss2=tan_loss(preds_dual, actual_dual)

        return 1. - 0.5*(loss1+loss2)

    '''


class ResUNet_Model:
    def __res_block(self, x, filters, kernel_size, strides, dilation_rate):
        def arm(dilation):
            batch_norm=tf.keras.layers.BatchNormalization()(x)
            relu=tf.keras.layers.ReLU()(batch_norm)
            conv=tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding='same')(relu)
            batch_norm=tf.keras.layers.BatchNormalization()(conv)
            relu=tf.keras.layers.ReLU()(batch_norm)
            conv=tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding='same')(relu)
            return conv

        outcomes=[x] # might cause problems with size...
        for dilation in dilation_rate:
            outcomes.append(arm(dilation))
        
        value=(tf.keras.layers.Add()(outcomes))

        return value
        

    def __psp_pooling(self, x, features):
        def unit(index, pool_size):
            # split= split into 1/4

            split=tf.split(value=x, num_or_size_splits=4, axis=3)[index] # axis=0 is the default, but not sure is correct... need dyn. index?

            max_pool=tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='same')(split)

            restore_dim=tf.keras.layers.UpSampling2D(size=pool_size, interpolation='bilinear')(max_pool)

            conv2dbn=self.__conv2d_bn(restore_dim, filters=(features//4), kernel_size=(1,1), strides=(1,1), dilation_rate=1)

            return conv2dbn

        pool_size=[(1,1), (2,2), (4,4), (8,8)]
        outcomes=[x]
        i=1
        while i <= 4:
            outcomes.append(unit((i-1), pool_size=pool_size[i-1]))
            i+=1

        concat=tf.keras.layers.Concatenate()(outcomes)
        conv2dbn=self.__conv2d_bn(concat, filters=features, kernel_size=(1,1), strides=(1,1), dilation_rate=1)
        return conv2dbn

    def __conv2d_bn(self, x, filters, kernel_size, strides, dilation_rate):
        conv=tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)(x)
        batch_norm=tf.keras.layers.BatchNormalization()(conv)
        return batch_norm

    def __combine(self, filters, x1, x2):
        relu=tf.keras.layers.ReLU()(x1)
        concat=tf.keras.layers.Concatenate()([relu, x2])
        conv2dbn=self.__conv2d_bn(concat, filters=filters, kernel_size=(1,1), strides=(1,1), dilation_rate=1)
        return conv2dbn

    def __upsample(self, x, filters):
        # double input (see figure 1 of paper)
        upsample=tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        conv2dbn=self.__conv2d_bn(upsample, filters=filters, kernel_size=(1,1), strides=(1,1), dilation_rate=1) # not sure about strides here...
        return conv2dbn

    def resunet_a_d6(self, in_width, in_height, in_channels, out_classes):
        inputs=Input(shape=(in_width, in_height, in_channels))
        # Encoder
        enc1=self.__conv2d_bn(inputs, filters=32, kernel_size=(1,1), strides=(1,1), dilation_rate=1)
        enc2=self.__res_block(enc1, filters=32, kernel_size=(3,3), strides=(1,1), dilation_rate=[1,3,15,31])
        enc3=tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=(2,2))(enc2)
        enc4=self.__res_block(enc3, filters=64, kernel_size=(3,3), strides=(1,1), dilation_rate=[1,3,15,31])
        enc5=tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(2,2))(enc4)
        enc6=self.__res_block(enc5, filters=128, kernel_size=(3,3), strides=(1,1), dilation_rate=[1,3,15])
        enc7=tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(2,2))(enc6)
        enc8=self.__res_block(enc7, filters=256, kernel_size=(3,3), strides=(1,1), dilation_rate=[1,3,15])
        enc9=tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(2,2))(enc8)
        enc10=self.__res_block(enc9, filters=512, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        enc11=tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(2,2))(enc10)
        enc12=self.__res_block(enc11, filters=1024, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        # Pooling bridge
        psp=self.__psp_pooling(enc12, features=1024)
        # Decoder
        dec14=self.__upsample(psp, filters=512)
        dec15=self.__combine(filters=512, x1=dec14, x2=enc10)
        dec16=self.__res_block(dec15, filters=512, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        dec17=self.__upsample(dec16, filters=256)
        dec18=self.__combine(filters=256, x1=dec17, x2=enc8)
        dec19=self.__res_block(dec18, filters=256, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        dec20=self.__upsample(dec19, filters=128)
        dec21=self.__combine(filters=128, x1=dec20, x2=enc6)
        dec22=self.__res_block(dec21, filters=128, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        dec23=self.__upsample(dec22, filters=64)
        dec24=self.__combine(filters=64, x1=dec23, x2=enc4)
        dec25=self.__res_block(dec24, filters=64, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        dec26=self.__upsample(dec25, filters=32)
        dec27=self.__combine(filters=32, x1=dec26, x2=enc2)
        dec28=self.__res_block(dec27, filters=32, kernel_size=(3,3), strides=(1,1), dilation_rate=[1])
        dec29=self.__combine(filters=32, x1=dec28, x2=enc1)
        # PSP Pooling 
        psp=self.__psp_pooling(dec29, features=32)
        dec31=tf.keras.layers.Conv2D(filters=out_classes, kernel_size=(1,1), strides=(1,1), dilation_rate=1)(psp)
        dec32=tf.keras.layers.Softmax(axis=-1)(dec31) #changing from axis=1 to axis=-1 fixed this...
        # return model
        model=tf.keras.models.Model(inputs=inputs, outputs=dec32)
        return model

        

    
