import tensorflow as tf
import numpy as np




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
