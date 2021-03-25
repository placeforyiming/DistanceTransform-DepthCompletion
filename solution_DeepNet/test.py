import tensorflow as tf
import numpy as np
import csv
import os
import sys
import time
sys.path.append("../")
from net import *
from data_read import *
from data_read import *
from evaluation import *

import argparse


import time




parser = argparse.ArgumentParser()



parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="KITTI",help='KITTI, NYU')


parser.add_argument('--model_name', action="store", dest= "model_name",default="ResNet18_NO_BN_l2_image",help='ResNet18_NO_BN_l2,ResNet18_NO_BN_l2_light,ResNet18_NO_BN_l2_light_light,ResNet18_NO_BN_l2_image,ResNet18_NO_BN_l2_light_image,ResNet18_NO_BN_l2_light_light_image,Sparsity')


parser.add_argument('--epoch_eval', action="store", dest="epoch_eval", type=int, default=6,help='eval from which epoch')


parser.add_argument('--table_size', action="store", dest="table_size", type=int, default=7,help='the size of look up table')

parser.add_argument('--scale_range', action="store", dest="scale_range", type=int, default=90.0,help='scale range')

parser.add_argument('--scale_num', action="store", dest="scale_num", type=int, default=4,help='scale num')

parser.add_argument('--line_num', action="store", dest="line_num", type=int, default=64,help='how many lidar lines, refer to sample_num for NYU')

parser.add_argument('--correct', action="store", dest="correct", type=bool, default=True,help='if correct outliers')



input_parameters = parser.parse_args()




save_path='./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'

save_path=save_path+str(input_parameters.correct)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num)+'/'



if input_parameters.model_name=='Sparsity':
    depth_network=SparsityCNN(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)



if input_parameters.model_name=='ResNet18_NO_BN_l2':
    depth_network=ResNet18_NO_BN_l2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light':
    depth_network=ResNet18_NO_BN_l2_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_2':
    depth_network=ResNet18_NO_BN_l2_light_2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_4':
    depth_network=ResNet18_NO_BN_l2_light_4(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_8':
    depth_network=ResNet18_NO_BN_l2_light_8(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_light':
    depth_network=ResNet18_NO_BN_l2_light_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)



if input_parameters.model_name=='ResNet18_NO_BN_l2_image':
    depth_network=ResNet18_NO_BN_l2_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_image':
    depth_network=ResNet18_NO_BN_l2_light_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_light_image':
    depth_network=ResNet18_NO_BN_l2_light_light_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)






depth_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_eval)+"_full")




#validation
for i in range(1000):
    print (i)
    img_batch,lidar_batch,index= read_one_test(i)
    
    lidar_only=lidar_batch
    lidar_batch=np.expand_dims(lidar_batch,axis=-1)



    if input_parameters.dataset_name=='KITTI':
        lidar = lidar_batch[:,96:,:,:]
        
        rgb=img_batch[:,96:,:,:]/255.0
        rgb=np.asarray(rgb).astype(np.float32)
    if input_parameters.dataset_name=='NYU':
            
        rgb=img_batch/255.0
        rgb=np.asarray(rgb).astype(np.float32)



    if "image" in input_parameters.model_name:
        depth_predicted,lidar_correction=depth_network.call(lidar,rgb)      
    else:
        depth_predicted,lidar_correction=depth_network.call(lidar)


    depth_predicted=np.squeeze(tf.nn.relu(depth_predicted-0.9)+0.9)

    #depth_predicted=np.squeeze(depth_predicted)

    depth_predicted=tf.clip_by_value(depth_predicted,0.0,100.0)

    depth_predicted=np.squeeze(depth_predicted)

    extra_depth=depth_predicted[0,:]
    extra_depth=np.tile(extra_depth,(96,1)).astype(np.float32)
    depth_predicted=np.vstack((extra_depth,depth_predicted))
    
    depth_predicted=depth_predicted*256.0
    depth_predicted=depth_predicted.astype(np.uint16)
    im=Image.fromarray(depth_predicted)
    im.save("./output/"+index)


   
