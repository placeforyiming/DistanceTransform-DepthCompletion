import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append("../")





class SparsityCNN(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="SparsityCNN") as scope:
            super(SparsityCNN , self).__init__()
            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            #self.bn_image_conv = tf.keras.layers.BatchNormalization()
            
            self.conv1a = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            #self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            #self.bn_1b = tf.keras.layers.BatchNormalization()

            self.conv1c = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            #self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            #self.bn_1d = tf.keras.layers.BatchNormalization()

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="SparsityCNN") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask

            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)



            tensor_1=self.conv1a(lidar_conv_total)
            #tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            #tensor_1=self.bn_1b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1c(tensor_1) 
            
            #tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            #tensor_1=self.bn_1d(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            output=self.conv_output(tensor_1)
            

        return output*self.scale_range,lidar_correct*self.scale_range
            

class ResNet18_NO_BN_l2(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2") as scope:
            super(ResNet18_NO_BN_l2, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(256, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(256, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(128, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)



            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range
   
class ResNet18_NO_BN_l2_cross_domain(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_cross_domain") as scope:
            super(ResNet18_NO_BN_l2_cross_domain, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

        
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(256, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(256, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(128, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_cross_domain") as scope:


         


            tensor_1=self.conv1a(input_lidar/self.scale_range)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(tensor_1)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,output*self.scale_range
   


class ResNet18_NO_BN_l2_light(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light") as scope:
            super(ResNet18_NO_BN_l2_light, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
              
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            #lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(input_lidar,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)


            lidar_conv_total=self.conv1_pre(lidar_conv_total) 

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)
            
            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range

class ResNet18_NO_BN_l2_light_2(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_2") as scope:
            super(ResNet18_NO_BN_l2_light_2, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



            self.conv1_pre= tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
  
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_2") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            #lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(input_lidar,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)

            lidar_conv_total=self.conv1_pre(lidar_conv_total) 

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            
            
            
            
            tensor_up_3=self.upsample_3(tensor_3)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range

class ResNet18_NO_BN_l2_light_4(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_4") as scope:
            super(ResNet18_NO_BN_l2_light_4, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



            self.conv1_pre= tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_4") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            #lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(input_lidar,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)


            lidar_conv_total=self.conv1_pre(lidar_conv_total) 

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

       
            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            
            
            
                      
            tensor_up_2=self.upsample_2(tensor_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range

class ResNet18_NO_BN_l2_light_8(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_8") as scope:
            super(ResNet18_NO_BN_l2_light_8, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



            self.conv1_pre= tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_8") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            #lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(input_lidar,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)


            lidar_conv_total=self.conv1_pre(lidar_conv_total) 

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
    
            tensor_up_1=self.upsample_1(tensor_1)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range


class ResNet18_NO_BN_l2_light_8_extra(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_8_extra") as scope:
            super(ResNet18_NO_BN_l2_light_8_extra, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05



            self.conv1_pre_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1_pre_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1_pre_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1_pre_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_8_extra") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range


            lidar_conv_total=self.conv1_pre_1(input_lidar)

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

            lidar_conv_total=self.conv1_pre_2(lidar_conv_total)

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

            lidar_conv_total=self.conv1_pre_3(lidar_conv_total)

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

            lidar_conv_total=self.conv1_pre_4(lidar_conv_total)

            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_total)

                        
            

            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
    
            tensor_up_1=self.upsample_1(tensor_1)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,input_lidar*self.scale_range



class ResNet18_NO_BN_l2_light_light(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_light") as scope:
            super(ResNet18_NO_BN_l2_light_light, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            self.conv1a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(32, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(32, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(32, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_light") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask


            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)



            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range






class ResNet18_NO_BN_l2_image(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_image") as scope:
            super(ResNet18_NO_BN_l2_image, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.img_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(256, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(256, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(128, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar,input_rgb):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_image") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask

            
            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            img_conv_1=self.img_conv(input_rgb)

            img_conv_1=tf.nn.leaky_relu(img_conv_1)

            lidar_conv_total=tf.concat([ img_conv_1,lidar_conv_total],axis=-1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)



            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range
            

class ResNet18_NO_BN_l2_light_image(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_image") as scope:
            super(ResNet18_NO_BN_l2_light_image, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.img_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.lidar_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(16, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            #self.conv2_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            #self.conv3_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            #self.conv4_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(8, 8),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            

            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            
            self.conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv4b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar,input_rgb):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_light_image") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask

            
            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            img_conv_1=self.img_conv(input_rgb)

            img_conv_1=tf.nn.leaky_relu(img_conv_1)

            lidar_conv_total=lidar_conv_total


            #lidar_conv_total=tf.concat([ img_conv_1,lidar_conv_total],axis=-1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)

            lidar_conv_total=self.conv1_pre(lidar_conv_total)

            lidar_conv_total_1=tf.concat([img_conv_1,lidar_conv_total],axis=-1)

            tensor_1=self.conv1a(lidar_conv_total_1)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total_1)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)

            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
                        
            tensor_2_pre=tf.nn.max_pool(lidar_conv_total, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
            tensor_2_total=tf.concat([tensor_2_total,tensor_2_pre],axis=-1)

            tensor_3=self.conv3a(tensor_2_total)
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            tensor_3_pre=tf.nn.max_pool(lidar_conv_total, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],padding='SAME')

            tensor_3_total=tf.concat([tensor_3_total,tensor_3_pre],axis=-1)
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
                        
            tensor_4_pre=tf.nn.max_pool(lidar_conv_total, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],padding='SAME')

            tensor_4=tf.concat([tensor_4,tensor_4_pre],axis=-1)
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range
            


class ResNet18_NO_BN_l2_large_image(tf.keras.Model):
    def __init__(self,table_size,if_correct=True,scale_range=90,scale_num=4,joint_train=False):

        with tf.name_scope(name="ResNet18_NO_BN_l2_large_image") as scope:
            super(ResNet18_NO_BN_l2_large_image, self).__init__()



            self.table_size=table_size
            self.scale_range=scale_range
            self.scale_num=scale_num
            self.if_correct=if_correct
            self.joint_train=joint_train

            self.regu_term=0.05

            if self.if_correct:
                self.correct_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

                self.correct_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
     
            self.img_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.lidar_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.weights_matrix=self.create_weight_matrix()

            if self.scale_num>1:

              self.lidar_conv_extra_1 = tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>2:

              self.lidar_conv_extra_2 = tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            if self.scale_num>3:
                
              self.lidar_conv_extra_3 = tf.keras.layers.Conv2D(32, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=0.05))



              
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1a_extra1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1a_extra2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1c_extra1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1c_extra2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
           
            self.conv2a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            self.conv2a_extra1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv2a_extra2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv2b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv2a_extra = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv2c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))

            self.conv2c_extra1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv2c_extra2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv2d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            
            
            
            self.conv3a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv3a_extra1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv3a_extra2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.conv3a_extra = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
                        
            self.conv3c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv3c_extra1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv3c_extra2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
           
            self.conv3d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            


            self.conv4a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv4a_extra1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv4a_extra2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
           
            
            self.conv4b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
                        
            self.conv4a_extra = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.conv4c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv4c_extra1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            self.conv4c_extra2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
           
            
            self.conv4d = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



            
            
            

            self.upsample_4=tf.keras.layers.Conv2DTranspose(512, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_4_post=tf.keras.layers.Conv2D(512, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_3=tf.keras.layers.Conv2DTranspose(256, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_3_post=tf.keras.layers.Conv2D(256, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            
            self.upsample_2=tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_2_post=tf.keras.layers.Conv2D(128, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            
            self.upsample_1=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            
            self.upsample_1_post=tf.keras.layers.Conv2D(64, 3, strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))
            
            

            self.conv_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=self.regu_term))



    def create_weight_matrix(self):
        assert (self.table_size+1)%2==0
        middle=(self.table_size-1)/2
        weight_matrix=np.zeros((self.table_size,self.table_size))
        for i in range(self.table_size):
            for j in range(self.table_size):
                distance=(self.table_size-abs(i-middle)-abs(j-middle))
                weight_matrix[i,j]=distance

        weight_matrix=np.reshape(weight_matrix,(self.table_size*self.table_size,))
        return weight_matrix.astype(np.float32)

    def generate_multi_channel(self,lidar_data,lidar_mask):
        

        lidar_1=lidar_data
        if self.scale_num>1:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_2=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_2,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>2:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_3=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 
            lidar_data=tf.expand_dims(lidar_3,axis=-1)
            lidar_mask=tf.math.greater(lidar_data,0.001)
            lidar_mask=tf.dtypes.cast(lidar_mask,tf.float32)
        if self.scale_num>3:

            extracted_input=tf.image.extract_patches(lidar_data,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            extracted_mask=tf.image.extract_patches(lidar_mask,sizes=(1,self.table_size,self.table_size,1),strides=(1,1,1,1),rates=(1,1,1,1),padding='SAME')
            max_index=tf.math.equal(extracted_mask*self.weights_matrix,tf.math.reduce_max(extracted_mask*self.weights_matrix,axis=-1,keepdims=True))
            max_index=tf.dtypes.cast(max_index,tf.float32)
            lidar_4=tf.reduce_sum(extracted_input*max_index,axis=-1)/(0.000001+tf.reduce_sum(max_index,axis=-1)) 

        if self.scale_num==1:
            return lidar_1,None,None,None
        if self.scale_num==2:
            return lidar_1,lidar_2,None,None
        if self.scale_num==3:
            return lidar_1,lidar_2,lidar_3,None
        if self.scale_num==4:
            return lidar_1,lidar_2,lidar_3,lidar_4


    def call(self, input_lidar,input_rgb):
       

        with tf.name_scope(name="ResNet18_NO_BN_l2_large_image") as scope:


            valued_mask=tf.math.greater(input_lidar,0.1)
            valued_mask=tf.dtypes.cast(valued_mask,tf.float32)

            input_lidar=input_lidar/self.scale_range

            if self.if_correct:
                lidar_correct=self.correct_1(input_lidar)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_2(lidar_correct)

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_3(lidar_correct) 

                lidar_correct=tf.nn.leaky_relu(lidar_correct)

                lidar_correct=self.correct_4(lidar_correct) 
            else:
                lidar_correct=input_lidar

            lidar_correct=lidar_correct*valued_mask

            
            lidar_1,lidar_2,lidar_3,lidar_4=self.generate_multi_channel(lidar_correct,valued_mask)

            if not (self.joint_train):

                lidar_1=tf.stop_gradient(lidar_1)
                lidar_2=tf.stop_gradient(lidar_2)
                lidar_3=tf.stop_gradient(lidar_3)
                lidar_4=tf.stop_gradient(lidar_4)
                

            lidar_conv_1=self.lidar_conv(lidar_1)  
            
           
            lidar_conv_total=tf.nn.leaky_relu(lidar_conv_1)


            img_conv_1=self.img_conv(input_rgb)

            img_conv_1=tf.nn.leaky_relu(img_conv_1)

            lidar_conv_total=tf.concat([ img_conv_1,lidar_conv_total],axis=-1)


            if self.scale_num>1:


                lidar_conv_2= tf.expand_dims(lidar_2,axis=-1)


                lidar_conv_2=self.lidar_conv_extra_1(lidar_conv_2)              
           
            
                lidar_conv_2=tf.nn.leaky_relu(lidar_conv_2)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_2],axis=-1)


            if self.scale_num>2:


                lidar_conv_3= tf.expand_dims(lidar_3,axis=-1)


                lidar_conv_3=self.lidar_conv_extra_2(lidar_conv_3)              
           
            
                lidar_conv_3=tf.nn.leaky_relu(lidar_conv_3)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_3],axis=-1)


            if self.scale_num>3:


                lidar_conv_4= tf.expand_dims(lidar_4,axis=-1)


                lidar_conv_4=self.lidar_conv_extra_1(lidar_conv_4)              
           
            
                lidar_conv_4=tf.nn.leaky_relu(lidar_conv_4)


                lidar_conv_total=tf.concat([lidar_conv_total,lidar_conv_4],axis=-1)



            tensor_1=self.conv1a(lidar_conv_total)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1a_extra1(tensor_1)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1a_extra2(tensor_1)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            

            tensor_1=self.conv1b(tensor_1)

            tensor_1_add=self.conv1a_extra(lidar_conv_total)
            

            tensor_1_total=tensor_1+tensor_1_add
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            

            tensor_1=self.conv1c(tensor_1_total) 
            
            
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1c_extra1(tensor_1)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1c_extra2(tensor_1)

            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            
            tensor_1=self.conv1d(tensor_1)
            
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            
            tensor_2=self.conv2a(tensor_1_total)
            
           
            
            tensor_2=tf.nn.leaky_relu(tensor_2)

            tensor_2=self.conv2a_extra1(tensor_2)

            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2a_extra2(tensor_2)

            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            
            tensor_2=self.conv2b(tensor_2)
            
            
            
            tensor_2_add=self.conv2a_extra(tensor_1_total)
            
            
            
            tensor_2_total=tensor_2+tensor_2_add
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_2=self.conv2c(tensor_2_total) 
            
            
            
            tensor_2=tf.nn.leaky_relu(tensor_2)


            tensor_2=self.conv2c_extra1(tensor_2)

            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2c_extra2(tensor_2)

            tensor_2=tf.nn.leaky_relu(tensor_2)

            
            tensor_2=self.conv2d(tensor_2)
            
            
            
            tensor_2_total=tensor_2+tensor_2_total
            
            tensor_2_total=tf.nn.leaky_relu(tensor_2_total)
            
            
            tensor_3=self.conv3a(tensor_2_total)
                        
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3a_extra1(tensor_3)

            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3a_extra2(tensor_3)

            tensor_3=tf.nn.leaky_relu(tensor_3)
            

            
            tensor_3=self.conv3b(tensor_3)
            
            
            
            tensor_3_add=self.conv3a_extra(tensor_2_total)
            
            
            
            tensor_3_total=tensor_3+tensor_3_add
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
            tensor_3=self.conv3c(tensor_3_total) 
            
            
            
            tensor_3=tf.nn.leaky_relu(tensor_3)


            tensor_3=self.conv3c_extra1(tensor_3)

            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3c_extra2(tensor_3)

            tensor_3=tf.nn.leaky_relu(tensor_3)
            


            
            tensor_3=self.conv3d(tensor_3)
            
            
            tensor_3_total=tensor_3+tensor_3_total
            
            tensor_3_total=tf.nn.leaky_relu(tensor_3_total)
            
            
                                   
            tensor_4=self.conv4a(tensor_3_total)
            
            
            
            tensor_4=tf.nn.leaky_relu(tensor_4)

            tensor_4=self.conv4a_extra1(tensor_4)

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4a_extra2(tensor_4)

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            tensor_4=self.conv4b(tensor_4)
            
            
            
            tensor_4_add=self.conv4a_extra(tensor_3_total)
            
            
            
            tensor_4_total=tensor_4+tensor_4_add
            
            tensor_4_total=tf.nn.leaky_relu(tensor_4_total)
            
            
            tensor_4=self.conv4c(tensor_4_total) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            tensor_4=self.conv4c_extra1(tensor_4)

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4c_extra2(tensor_4)

            tensor_4=tf.nn.leaky_relu(tensor_4)

            tensor_4=self.conv4d(tensor_4) 

            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            
            
            
            
            tensor_up_4=self.upsample_4(tensor_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            tensor_up_4=tf.concat([tensor_up_4,tensor_3_total],axis=-1)
            
            tensor_up_4=self.upsample_4_post(tensor_up_4)
            
            
            
            tensor_up_4=tf.nn.leaky_relu(tensor_up_4)
            
            
            
            tensor_up_3=self.upsample_3(tensor_up_4)
            
            
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            tensor_up_3=tf.concat([tensor_up_3,tensor_2_total],axis=-1)
            
            tensor_up_3=self.upsample_3_post(tensor_up_3)
            
           
            
            tensor_up_3=tf.nn.leaky_relu(tensor_up_3)
            
            
                      
            tensor_up_2=self.upsample_2(tensor_up_3)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            tensor_up_2=tf.concat([tensor_up_2,tensor_1_total],axis=-1)
            
            tensor_up_2=self.upsample_2_post(tensor_up_2)
            
            
            
            tensor_up_2=tf.nn.leaky_relu(tensor_up_2)
            
            
            
            tensor_up_1=self.upsample_1(tensor_up_2)
            
            
            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            tensor_up_1=tf.concat([tensor_up_1,lidar_conv_total],axis=-1)
            
            tensor_up_1=self.upsample_1_post(tensor_up_1)
            

            
            tensor_up_1=tf.nn.leaky_relu(tensor_up_1)
            
            output=self.conv_output(tensor_up_1)


        return output*self.scale_range,lidar_correct*self.scale_range
  