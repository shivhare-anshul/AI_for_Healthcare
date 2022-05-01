import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Add, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import merge
import cv2
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from keras.utils.vis_utils import plot_model
import math
from scipy import ndimage
from PIL import Image
import glob
import pandas as pd

tf.config.run_functions_eagerly(True)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # or '0,1', or '0,1,2' or '1,2,3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')

print('Visible Physical Devices: ',physical_devices)

for gpu in physical_devices:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.threading.set_inter_op_parallelism_threads(32)

tf.config.threading.set_intra_op_parallelism_threads(32)

def dncnn():
    layer_in = Input(shape=(256,256,3))
    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(layer_in)
    #D=7
    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    x=Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)

    out=Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='relu',use_bias=False)(x)

    model = Model(inputs=layer_in, outputs=out)

    return model

def psnr(y_true,y_pred):
    #y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    #y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.psnr(y_true,y_pred,255)

def ssim(y_true,y_pred):
    #y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    #y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.ssim(y_true,y_pred,255)

def ssim_loss(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0) * 3)


def Denoiser_DnCNN(input_image_path,output_path):
    model = dncnn()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=["mean_squared_logarithmic_error", ssim_loss], optimizer=opt, metrics=[psnr,ssim])
    model.load_weights('/home/anshul/Anshul_flask/Anshul_Flask/Bscan_perceptual_ssim_mse_cp.ckpt')
    


    img__=cv2.imread(input_image_path)
    img__=cv2.resize(img__,(256,256),interpolation = cv2.INTER_AREA)
    #img_unorm = img__*255
    #plt.imshow(img__)
    im = Image.fromarray(img__)
    input_model = np.expand_dims(img__, axis=0)
    s= model.predict(input_model,verbose=2)
    s= np.squeeze(s, 0)
    im1=Image.fromarray((s.astype('uint8')))
    file = os.path.split(input_image_path)[1].split('.')[0]
    output_path_pred = output_path + file + '_pred.jpg'
    output_path_ip = output_path + file + '.jpg'
    im1.save(output_path_pred)
    cv2.imwrite(output_path_ip,img__)
    #img_unorm.save(output_path_ip)

    psnr1=tf.image.psnr(img__,im1,255)
    

    return {'file':file, 'psnr':0.5}

def Thresholding(input_image_path,output_path):

    #image_path=fold_path+df['image_path'][i]

    file = os.path.split(input_image_path)[1].split('.')[0]
    output_path_pred = output_path + file + '_pred.jpg'
    output_path_ip = output_path + file + '.jpg'

    curr_img=cv2.imread(input_image_path)
    cv2.imwrite(output_path_ip,curr_img)
    curr_img=2.0*np.sqrt(curr_img+ 3.0/8.0)
    curr_img=(curr_img-np.average(curr_img))/np.max(curr_img)
    curr_img=np.where(curr_img<=0,0,curr_img)
    curr_img=(curr_img*255)
    curr_img=curr_img.astype(np.uint8)
    cv2.imwrite(output_path_pred,curr_img)

    return {'file':file, 'psnr':0.5}


    


# path = 'Bscan_1.jpg'
# outpath1 = 'output1.jpg'
# outpath2 = 'output2.jpg'
# infer_sar(path, outpath1, outpath2)