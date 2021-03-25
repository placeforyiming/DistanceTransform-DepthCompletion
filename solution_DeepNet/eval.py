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


parser.add_argument('--model_name', action="store", dest= "model_name",default="ResNet18_NO_BN_l2_light_image",help='ResNet18_NO_BN_l2,ResNet18_NO_BN_l2_cross_domain,ResNet18_NO_BN_l2_light,ResNet18_NO_BN_l2_light_light,ResNet18_NO_BN_l2_image,ResNet18_NO_BN_l2_light_image,Sparsity')


parser.add_argument('--epoch_eval', action="store", dest="epoch_eval", type=int, default=7,help='eval from which epoch')


parser.add_argument('--table_size', action="store", dest="table_size", type=int, default=7,help='the size of look up table')

parser.add_argument('--scale_range', action="store", dest="scale_range", type=int, default=90.0,help='scale range')

parser.add_argument('--scale_num', action="store", dest="scale_num", type=int, default=4,help='scale num')

parser.add_argument('--line_num', action="store", dest="line_num", type=int, default=64,help='how many lidar lines, refer to sample_num for NYU')

parser.add_argument('--correct', action="store", dest="correct", type=bool, default=True,help='if correct outliers')


input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

save_path='./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'

save_path=save_path+str(input_parameters.correct)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num)+'/'


if input_parameters.model_name=='Sparsity':
    depth_network=SparsityCNN(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)



if input_parameters.model_name=='ResNet18_NO_BN_l2':
    depth_network=ResNet18_NO_BN_l2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_cross_domain':
    depth_network=ResNet18_NO_BN_l2_cross_domain(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light':
    depth_network=ResNet18_NO_BN_l2_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_2':
    depth_network=ResNet18_NO_BN_l2_light_2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_4':
    depth_network=ResNet18_NO_BN_l2_light_4(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_8':
    depth_network=ResNet18_NO_BN_l2_light_8(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_8_extra':
    depth_network=ResNet18_NO_BN_l2_light_8_extra(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)



if input_parameters.model_name=='ResNet18_NO_BN_l2_light_light':
    depth_network=ResNet18_NO_BN_l2_light_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)



if input_parameters.model_name=='ResNet18_NO_BN_l2_image':
    depth_network=ResNet18_NO_BN_l2_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_image':
    depth_network=ResNet18_NO_BN_l2_light_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)

if input_parameters.model_name=='ResNet18_NO_BN_l2_large_image':
    depth_network=ResNet18_NO_BN_l2_large_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)




depth_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_eval)+"_full")

if input_parameters.dataset_name=='KITTI':
	evaluate=Result()
        

if input_parameters.dataset_name=='NYU':
	evaluate=Result_NYU()
    
        
rmse_total=0
mae_total=0
irmse_total=0
imae_total=0
if input_parameters.dataset_name=='NYU':
	delta_1_total=0.0
	delta_2_total=0.0
	delta_3_total=0.0
#validation


time_interval=[]
for i in range(1000):
    print (i)
    if i==654 and input_parameters.dataset_name=='NYU':
    	break

    if input_parameters.dataset_name=='KITTI':
        img_batch,lidar_batch,ground_truth,semantic= read_one_val(i,line_number=input_parameters.line_num)

    if input_parameters.dataset_name=='NYU':
    
        img_batch,lidar_batch,ground_truth=read_one_val_NYU(i,sample_rate=input_parameters.line_num)

    
    lidar_only=lidar_batch
    lidar_batch=np.expand_dims(lidar_batch,axis=-1)



    if input_parameters.dataset_name=='KITTI':
        lidar = lidar_batch[:,96:,:,:]
        gt = ground_truth[:,96:,:]
        rgb=img_batch[:,96:,:,:]/255.0
        rgb=np.asarray(rgb).astype(np.float32)
    if input_parameters.dataset_name=='NYU':
        gt = ground_truth
        lidar = lidar_batch
        rgb=img_batch/255.0
        rgb=np.asarray(rgb).astype(np.float32)



   

   


    if i==1:
    	total_parameters=0
    	all_variables=depth_network.trainable_variables
    	for mm in all_variables:
    		if len(np.shape(mm))>1:
    			total_parameters+=np.shape(mm)[0]*np.shape(mm)[1]*np.shape(mm)[2]*np.shape(mm)[3]
    		else:
    			total_parameters+=np.shape(mm)[0]
    	print (total_parameters)

    
    
    if i>20:
        time_a=time.time()
    if "image" in input_parameters.model_name:
        depth_predicted,lidar_correction=depth_network.call(lidar,rgb)      
    else:
        depth_predicted,lidar_correction=depth_network.call(lidar)

        '''
        depth_predicted=lidar_correction
        depth_predicted=np.squeeze(depth_predicted) 
        gt=np.squeeze(gt)
        final_mask=np.logical_and(depth_predicted>0.1,gt>0.1)
        depth_predicted=depth_predicted*final_mask
        gt=gt*final_mask
        '''



    if i>20:
    	time_b=time.time()
    	time_interval.append(time_b-time_a)

    if input_parameters.dataset_name=='NYU':
    	depth_predicted=np.squeeze(depth_predicted)[6:234,8:312]
    	gt=np.squeeze(gt)[6:234,8:312]
    else:
    	depth_predicted=np.squeeze(tf.nn.relu(depth_predicted-0.9)+0.9)
    evaluate.evaluate(np.squeeze(depth_predicted),np.squeeze(gt))
    irmse, imae, mse, rmse, mae=evaluate.irmse, evaluate.imae, evaluate.mse, evaluate.rmse, evaluate.mae
    if input_parameters.dataset_name=='NYU':
    	delta_1,delta_2,delta_3=evaluate.delta1,evaluate.delta2,evaluate.delta3
    	delta_1_total+=delta_1
    	delta_2_total+=delta_2
    	delta_3_total+=delta_3


    rmse_total+=rmse
    mae_total+=mae
    irmse_total+=irmse
    imae_total+=imae
    print ("rmse:")
    print (rmse_total/(i+1))
    print ("mae:")
    print (mae_total/(1+i))
    print ("irmse:")
    print (irmse_total/(1+i))
    print ("imae:")
    print (imae_total/(1+i))
    print ('method time')
    print (np.sum(time_interval)/len(time_interval))

          

if input_parameters.dataset_name=='NYU':
    print ("rmse:")
    print (rmse_total/654.0)
    print ("mae:")
    print (mae_total/654.0)      
    print ("irmse:")
    print (irmse_total/654.0)      
    print ("imae:")
    print (imae_total/654.0) 
    print (delta_1_total/654.0)
    print (delta_2_total/654.0)
    print (delta_3_total/654.0)
else:

	print ("rmse:")
	print (rmse_total/1000.0)
	print ("mae:")
	print (mae_total/1000.0)      
	print ("irmse:")
	print (irmse_total/1000.0)      
	print ("imae:")
	print (imae_total/1000.0)  


print ('method time')
print (np.sum(time_interval)/len(time_interval))


