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


parser.add_argument('--model_name', action="store", dest= "model_name",default="ResNet18_NO_BN_l2_image",help='ResNet18_NO_BN_l2,ResNet18_NO_BN_l2_image,Sparsity,Sparsity_image')


parser.add_argument('--epoch_eval', action="store", dest="epoch_eval", type=int, default=10,help='how many epochs to eval')

parser.add_argument('--table_size', action="store", dest="table_size", type=int, default=7,help='the size of look up table')


parser.add_argument('--scale_num', action="store", dest="scale_num", type=int, default=4,help='scale num')

parser.add_argument('--scale_range', action="store", dest="scale_range", type=int, default=90.0,help='scale range')


parser.add_argument('--line_num', action="store", dest="line_num", type=int, default=64,help='how many lidar lines, refer to sample_num for NYU')

parser.add_argument('--removal', action="store", dest="removal", type=bool, default=False,help='if remove outliers')

parser.add_argument('--model_type', action="store", dest="model_type", type=str, default="DT",help='baseline or DT or DT_NN')



input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def create_weight_matrix(size=11):
    assert (size+1)%2==0
    middle=(size-1)/2
    weight_matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            distance=10**(size-abs(i-middle)-abs(j-middle))
            weight_matrix[i,j]=distance

    weight_matrix=np.reshape(weight_matrix,(size*size,))
    return weight_matrix.astype(np.float32)


def nearest_point(refined_lidar):   
    value_mask=np.asarray(1.0-np.squeeze(refined_lidar)>0.1).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return dt,lbl


def DT_complete_batch(lidar_batch):
    batch_size=np.shape(lidar_batch)[0]
    new_batch=[]
    
    for i in range(batch_size):

        lidar_single=lidar_batch[i,:,:,0]
                
        dt,lbl=nearest_point(lidar_single)
        with_value=np.squeeze(lidar_single)>0.1

        depth_list=np.squeeze(lidar_single)[with_value]
        label_list=np.reshape(lbl,[1,1216*352])
        depth_list_all=depth_list[label_list-1]       
        depth_map=np.reshape(depth_list_all,(352,1216))
        new_batch.append(depth_map)   

    new_batch=np.asarray(new_batch)
    new_batch=np.expand_dims(new_batch,axis=-1)

    refined_lidar=new_batch.astype(np.float32)

    return refined_lidar

def generate_multi_channel(lidar_data,table_size,scale_range=90.0,scale_num=4):
    height,width=np.shape(np.squeeze(lidar_data))

    weights_matrix=create_weight_matrix(size=table_size)
    '''

    extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
    max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
    lidar_1=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
    lidar_data=tf.expand_dims(lidar_1,axis=-1)
    '''

    lidar_1=lidar_data[:,:,:,0]

    if scale_num>1:

        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_2=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
        lidar_data=tf.expand_dims(lidar_2,axis=-1)

    if scale_num>2:

        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_3=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
        lidar_data=tf.expand_dims(lidar_3,axis=-1)
    
    if scale_num>3:

        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_4=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))

    if scale_num==1:
        return lidar_1/scale_range,None,None,None
    if scale_num==2:
        return lidar_1/scale_range,lidar_2/scale_range,None,None
    if scale_num==3:
        return lidar_1/scale_range,lidar_2/scale_range,lidar_3/scale_range,None
    if scale_num==4:
        return lidar_1/scale_range,lidar_2/scale_range,lidar_3/scale_range,lidar_4/scale_range
   
def generate_multi_channel_with_image(rgb_data,lidar_data,table_size,scale_range=90.0,scale_num=4):
    height,width=np.shape(np.squeeze(lidar_data))
    weights_matrix=create_weight_matrix(size=table_size)

    '''
    extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
    max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
    lidar_1=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
    lidar_data=tf.expand_dims(lidar_1,axis=-1)
    lidar_1=tf.expand_dims(lidar_1,axis=-1)/scale_range
    '''

    lidar_1=lidar_data/scale_range
    lidar_1=tf.concat([rgb_data, lidar_1], 3)

    if scale_num>1:

        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_2=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
        lidar_data=tf.expand_dims(lidar_2,axis=-1)
        lidar_2=tf.expand_dims(lidar_2,axis=-1)/scale_range
        lidar_2=tf.concat([rgb_data, lidar_2], 3)

    if scale_num>2:
        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_3=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))
        lidar_data=tf.expand_dims(lidar_3,axis=-1)   
        lidar_3=tf.expand_dims(lidar_3,axis=-1)/scale_range
        lidar_3=tf.concat([rgb_data, lidar_3], 3)

    if scale_num>3:

        extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,table_size,table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
        max_index=tf.math.equal(extracted_input*weights_matrix,tf.math.reduce_max(extracted_input*weights_matrix,axis=-1,keepdims=True))
        lidar_4=tf.reduce_sum(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1)/(0.000001+tf.dtypes.cast(tf.math.count_nonzero(extracted_input*tf.dtypes.cast(max_index,tf.float32),axis=-1),tf.float32))    
        lidar_4=tf.expand_dims(lidar_4,axis=-1)/scale_range
        lidar_4=tf.concat([rgb_data, lidar_4], 3)

    if scale_num==1:
        return lidar_1/scale_range,None,None,None
    if scale_num==2:
        return lidar_1/scale_range,lidar_2/scale_range,None,None
    if scale_num==3:
        return lidar_1/scale_range,lidar_2/scale_range,lidar_3/scale_range,None
    if scale_num==4:
        return lidar_1/scale_range,lidar_2/scale_range,lidar_3/scale_range,lidar_4/scale_range


save_path='./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+input_parameters.model_type+'/'+str(input_parameters.line_num)+'/'


if not(input_parameters.model_type=='baseline'):
    save_path=save_path+str(input_parameters.removal)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num)+'/'




if input_parameters.model_name=='Sparsity':
    if input_parameters.model_type=='baseline'or input_parameters.model_type=="DT_NN":
        depth_network=SparsityCNN(scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)
    else:
        depth_network=SparsityCNN(if_baseline=False,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)


if input_parameters.model_name=='Sparsity_image':
    if input_parameters.model_type=='baseline'or input_parameters.model_type=="DT_NN":
        depth_network=SparsityCNN_image(scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)
    else:
        depth_network=SparsityCNN_image(if_baseline=False,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)



if input_parameters.model_name=='ResNet18_NO_BN_l2':
    if input_parameters.model_type=='baseline'or input_parameters.model_type=="DT_NN":
        depth_network=ResNet18_NO_BN_l2(scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)
    else:
        depth_network=ResNet18_NO_BN_l2(if_baseline=False,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)


if input_parameters.model_name=='ResNet18_NO_BN_l2_image':
    if input_parameters.model_type=='baseline'or input_parameters.model_type=="DT_NN":
        depth_network=ResNet18_NO_BN_l2_image(scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)
    else:
        depth_network=ResNet18_NO_BN_l2_image(if_baseline=False,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num)





    #load weights
if not(input_parameters.model_name=='ResNet34_large_NO_BN_l2'):
    depth_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_eval)+"_full")



if input_parameters.dataset_name=='KITTI':
	data_loader=KITTI_demo_loader()
        

if input_parameters.dataset_name=='NYU':
	data_loader=NYU_demo_loader()
    
        
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
        img_batch,lidar_batch,save_name= data_loader.read_one_image()

    if input_parameters.dataset_name=='NYU':
    
        img_batch,lidar_batch,save_name=data_loader.read_one_image()

    
    lidar_only=np.squeeze(lidar_batch)
    lidar_batch=np.expand_dims(lidar_only,axis=-1)
    lidar_batch=np.expand_dims(lidar_batch,axis=0)
    img_batch=np.squeeze(img_batch)
    img_batch=np.expand_dims(img_batch,axis=0)



    if input_parameters.model_type=='DT_NN':
        lidar_batch=DT_complete_batch(lidar_batch)


    if input_parameters.dataset_name=='KITTI':
        lidar = lidar_batch[:,96:,:,:]

        rgb=img_batch[:,96:,:,:]/255.0
        rgb=np.asarray(rgb).astype(np.float32)
    if input_parameters.dataset_name=='NYU':

        lidar = lidar_batch
        rgb=img_batch/255.0
        rgb=np.asarray(rgb).astype(np.float32)



    if input_parameters.model_type=="DT":
        if_baseline=False
        if "_image" in input_parameters.model_name:
            lidar_1,lidar_2,lidar_3,lidar_4=generate_multi_channel_with_image(rgb,lidar,input_parameters.table_size,input_parameters.scale_range,input_parameters.scale_num)

        else:

            lidar_1,lidar_2,lidar_3,lidar_4=generate_multi_channel(lidar,input_parameters.table_size,input_parameters.scale_range,input_parameters.scale_num)
    else:
        lidar=np.squeeze(lidar)
        lidar=np.expand_dims(lidar,axis=0) 

   


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
    if input_parameters.model_type=="baseline" or input_parameters.model_type=='DT_NN':
        depth_predicted=depth_network.call(lidar)
    else:
        depth_predicted=depth_network.call(lidar_1,lidar_2,lidar_3,lidar_4)
    depth_predicted=tf.nn.relu(depth_predicted)


    if input_parameters.dataset_name=='KITTI':

        img_total=np.concatenate([np.squeeze(rgb),np.squeeze(np.tile(lidar/100.0,(1,1,1,3)))],axis=0)
        img_total=np.concatenate([np.squeeze(img_total),np.squeeze(np.tile(depth_predicted/100.0,(1,1,1,3)))],axis=0)
       
        plt.imsave('./demo/KITTI/'+save_name,np.squeeze(img_total))


        

    if input_parameters.dataset_name=='NYU':

        img_total=np.concatenate([np.squeeze(rgb),np.squeeze(np.tile(lidar/np.max(lidar),(1,1,1,3)))],axis=0)
        img_total=np.concatenate([np.squeeze(img_total),np.squeeze(np.tile(depth_predicted/np.max(depth_predicted),(1,1,1,3)))],axis=0)
       
        plt.imsave('./demo/NYU/'+save_name[:-3]+'.png',np.squeeze(img_total))

    
   


