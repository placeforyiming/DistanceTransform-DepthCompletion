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
#from evaluation import *


import argparse







parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="KITTI",help='KITTI, NYU')


parser.add_argument('--model_name', action="store", dest= "model_name",default="ResNet18_NO_BN_l2_light_8_extra",help='ResNet18_NO_BN_l2,ResNet18_NO_BN_l2_cross_domain,ResNet18_NO_BN_l2_light,ResNet18_NO_BN_l2_light_2,ResNet18_NO_BN_l2_light_4,ResNet18_NO_BN_l2_light_8,ResNet18_NO_BN_l2_light_light,ResNet18_NO_BN_l2_image,ResNet18_NO_BN_l2_light_image,Sparsity')

parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.0001,help='learning_rate')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=2,help='batch_size')

parser.add_argument('--learning_rate_decay_f', action="store", dest="learning_rate_decay_f", type=float, default=2,help='how many epochs to over lr by 2')

parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=4000,help='save and eval after how many iterations')

parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=1,help='train from which epoch')

parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=8,help='how many epochs to train')

parser.add_argument('--table_size', action="store", dest="table_size", type=int, default=7,help='the size of look up table')

parser.add_argument('--scale_range', action="store", dest="scale_range", type=int, default=90.0,help='scale range')

parser.add_argument('--scale_num', action="store", dest="scale_num", type=int, default=4,help='scale num')

parser.add_argument('--line_num', action="store", dest="line_num", type=int, default=64,help='how many lidar lines, refer to sample_num for NYU')

parser.add_argument('--correct', action="store", dest="correct", type=bool, default=False,help='if correct outliers')

parser.add_argument('--joint_train', action="store", dest="joint_train", type=bool, default=False,help='if joint_train')




input_parameters = parser.parse_args()


'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''




save_path='./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'





if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name)

if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num))):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num))

save_path=save_path+str(input_parameters.correct)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num)+'/'
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct))):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct))
if not (os.path.exists('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct)+'/'+str(input_parameters.table_size))):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct)+'/'+str(input_parameters.table_size))
if not (os.path.exists('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num))):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name+'/'+input_parameters.model_name+'/'+str(input_parameters.line_num)+'/'+str(input_parameters.correct)+'/'+str(input_parameters.table_size)+'/'+str(input_parameters.scale_num))


lr=input_parameters.learning_rate



if input_parameters.model_name=='Sparsity':
    depth_network=SparsityCNN(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)



if input_parameters.model_name=='ResNet18_NO_BN_l2':
    depth_network=ResNet18_NO_BN_l2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_cross_domain':
    depth_network=ResNet18_NO_BN_l2_cross_domain(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light':
    depth_network=ResNet18_NO_BN_l2_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_2':
    depth_network=ResNet18_NO_BN_l2_light_2(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_4':
    depth_network=ResNet18_NO_BN_l2_light_4(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_8':
    depth_network=ResNet18_NO_BN_l2_light_8(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_8_extra':
    depth_network=ResNet18_NO_BN_l2_light_8_extra(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)


if input_parameters.model_name=='ResNet18_NO_BN_l2_light_light':
    depth_network=ResNet18_NO_BN_l2_light_light(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)



if input_parameters.model_name=='ResNet18_NO_BN_l2_image':
    depth_network=ResNet18_NO_BN_l2_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_light_image':
    depth_network=ResNet18_NO_BN_l2_light_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)

if input_parameters.model_name=='ResNet18_NO_BN_l2_large_image':
    depth_network=ResNet18_NO_BN_l2_large_image(table_size=input_parameters.table_size,if_correct=input_parameters.correct,scale_range=input_parameters.scale_range,scale_num=input_parameters.scale_num,joint_train=input_parameters.joint_train)





if input_parameters.epoch_start>1:
    #load weights
    depth_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"_full")


file = open(save_path+"record.txt","w") 
  

for current_epoch in range(input_parameters.epoch_num):


    if current_epoch>0 and current_epoch%input_parameters.learning_rate_decay_f==0:
        lr=lr/2.0
    
    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    if input_parameters.dataset_name=='KITTI':
    
        if input_parameters.line_num==64:
            Data_loader=Data_load(input_line=64)
        if input_parameters.line_num==32:
            Data_loader=Data_load(input_line=32)
        if input_parameters.line_num==16:
            Data_loader=Data_load(input_line=16)

    if input_parameters.dataset_name=='NYU':
    
        if input_parameters.line_num==20:
            Data_loader=Data_load_NYU(sample_rate=20)
        if input_parameters.line_num==50:
            Data_loader=Data_load_NYU(sample_rate=50)
        if input_parameters.line_num==200:
            Data_loader=Data_load_NYU(sample_rate=200)
        if input_parameters.line_num==500:
            Data_loader=Data_load_NYU(sample_rate=500)



    print (Data_loader.total_sample)

    print("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    print("Learning rate is " + str(lr)) 
    file.write("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    file.write('\n')
    file.write("Learning rate is " + str(lr))
    file.write('\n')
    error_ave_1000=0.0
    auxi_loss_total=0.0
    for iters in range(1000000):

        lidar,gt,rgb=Data_loader.read_batch(input_parameters.batch_size)
        if len(np.shape(gt))==1:
            depth_network.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"_full")
            break
        
        if input_parameters.dataset_name=='KITTI':
            lidar = lidar[:,96:,:,:]
            gt = gt[:,96:,:,:]
            rgb=rgb[:,96:,:,:]/255.0
            rgb=np.asarray(rgb).astype(np.float32)
            with_gt=(gt>0.1)
            with_input=(lidar>0.1)
        if input_parameters.dataset_name=='NYU':
            rgb=rgb/255.0
            rgb=np.asarray(rgb).astype(np.float32)
            with_gt=(gt>0.0001)
            with_input=(lidar>0.001)
        

        total_value=np.sum(with_gt)
        with_gt=tf.dtypes.cast(with_gt,tf.float32)

        with_input=np.logical_and(with_gt,with_input)
        total_value_input=np.sum(with_input)
        with_input=tf.dtypes.cast(with_input,tf.float32)


        with tf.GradientTape() as tape:

            if "image" in input_parameters.model_name:
                depth_predicted,lidar_correction=depth_network.call(lidar,rgb)
            else:
                depth_predicted,lidar_correction=depth_network.call(lidar)
          

            total_loss=(depth_predicted- gt)**2*with_gt
            if input_parameters.dataset_name=='NYU':
                total_loss=tf.math.sqrt(tf.reduce_sum(total_loss[:,6:228,8:304,:])/total_value)
            else:
                total_loss=tf.reduce_sum(total_loss)/total_value
            total_loss_record=total_loss

            if input_parameters.correct:
                auxi_loss=(lidar_correction- gt)**2*with_input+tf.math.abs(lidar_correction- gt)*with_input
                auxi_loss=tf.reduce_sum(auxi_loss)/total_value_input
                auxi_loss_total=auxi_loss_total+auxi_loss
                total_loss=total_loss+auxi_loss

        grads = tape.gradient(total_loss, depth_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, depth_network.trainable_variables))



        error_ave_1000=error_ave_1000+total_loss_record

        if iters%1000==0 :
            file.write(str(iters))
            file.write('\n')
            file.write(str(error_ave_1000/1000)) 
            file.write('\n')
            print (time.time())
            print(iters)
            print (save_path)
            print (error_ave_1000/1000)
            print (auxi_loss_total/1000)
            auxi_loss_total=0.0
            error_ave_1000=0.0

        if iters%input_parameters.save_eval_f==0 and iters>0:

            depth_network.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"_"+str(iters))

           
        lidar = None
        gt = None
        final_input = None
        img_batch = None
        lidar_batch = None
        ground_truth = None
        semantic = None
        depth_predicted = None
        evaluate=None
        with_gt=None
        total_value=None




file.close()
