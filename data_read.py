import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import tensorflow as tf
import cv2
import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

import glob
import ntpath

import skimage.io as io
import scipy.io as sio

from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
sys.path.append("../")

sys.path.append("../")


def get_paths_and_transform(num_line=64):
    if num_line==64:
        root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar')
    if num_line==32:
    	root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar_32')
    if num_line==16:
    	root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar_16')
    root_rgb = os.path.join('../depth_selection/KITTI/RGB')
    glob_sparse_lidar = "train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png"

    glob_sparse_lidar = os.path.join(root_d,glob_sparse_lidar)
    all_lidar_path_with_new=glob.glob(glob_sparse_lidar)
    all_lidar_path_without_new=[i for i in all_lidar_path_with_new if not (('left' in i) or('right' in i)) ]
    paths_sparse_lidar = sorted(all_lidar_path_without_new)
    def get_rgb_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
        return pnew
    
    glob_rgb = [get_rgb_paths(i) for i in paths_sparse_lidar]
    return paths_sparse_lidar,glob_rgb


def img_path_to_lidar(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'Sparse_Lidar/'+path_list[4]+'/'+path_list[5]+'/proj_depth/velodyne_raw/'+path_list[6]+'/'+path_list[8]

def img_path_to_ground_truth(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'ground_truth/'+path_list[4]+'/'+path_list[5]+'/proj_depth/groundtruth/'+path_list[6]+'/'+path_list[8]



    
    
def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    rgb_png =np.array(Image.fromarray(rgb_png).resize((1216,352), Image.NEAREST))
    img_file.close()
    return rgb_png

def depth_new_read(filename):
    depth=io.imread(filename)
    depth=depth/255.0*100
    return depth


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.

    depth  = np.array(Image.fromarray(depth).resize((1216,352), Image.NEAREST))

    depth = np.expand_dims(depth,-1)
    return depth


    
def outlier_removal(lidar):
    DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
    
    sparse_lidar=np.squeeze(lidar)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)
    
    
    lidar_sum=cv2.filter2D(sparse_lidar,-1,DIAMOND_KERNEL_7)
    
    lidar_count=cv2.filter2D(valid_pixels,-1,DIAMOND_KERNEL_7)
    
    lidar_aveg=lidar_sum/(lidar_count+0.00001)
    
    potential_outliers=((sparse_lidar-lidar_aveg)>1.0).astype(np.float)
    
    
    return  (sparse_lidar*(1-potential_outliers)).astype(np.float32)


lidar_path,img_path=get_paths_and_transform()
total_img=len(img_path)
total_lidar=len(lidar_path)





class Data_load():
    def __init__(self,input_line=64):
        if input_line==64:
        	lidar_path,img_path=get_paths_and_transform(64)
        if input_line==32:
        	lidar_path,img_path=get_paths_and_transform(32)
        if input_line==16:
        	lidar_path,img_path=get_paths_and_transform(16)
        self.lidar_path=lidar_path
        self.img_path=img_path
        self.num_sample=[i for i in range(len(self.img_path))]
        np.random.shuffle(self.num_sample)
        self.index=0
        self.total_sample=len(self.img_path)
       
    def read_batch(self,batch_size=4,if_removal=False):
        i=0
        img_batch=[]
        lidar_batch=[]
        gt_batch=[]

        while (i<(batch_size)):
            i=i+1
            
            img=rgb_read(self.img_path[self.num_sample[self.index]])


            depth=depth_read(self.lidar_path[self.num_sample[self.index]])
            
            if if_removal:
                depth=outlier_removal(depth)

            gt_path=img_path_to_ground_truth(self.img_path[self.num_sample[self.index]])
            ground_truth=depth_read(gt_path)

            lidar_batch.append(depth)
            img_batch.append(img)
            gt_batch.append(ground_truth)
            self.index=self.index+1
        if self.index+batch_size>self.total_sample:
            return [0],[1],[2]
        else:
            return  np.asarray(lidar_batch),np.asarray(gt_batch),np.asarray(img_batch)

        
def read_one_val(index,line_number=64,with_semantic=True,if_removal=False):
    ground_truth_path='../depth_selection/val_selection_cropped/groundtruth_depth'
    if line_number==64:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw'
    if line_number==32:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw_32'
    if line_number==16:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw_16'

    image_path='../depth_selection/val_selection_cropped/image'
    ground_truth=os.listdir('../depth_selection/val_selection_cropped/groundtruth_depth')
    image=os.listdir('../depth_selection/val_selection_cropped/image')
    velodyne_raw=os.listdir('../depth_selection/val_selection_cropped/velodyne_raw')
    intrinsics=os.listdir('../depth_selection/val_selection_cropped/intrinsics')
    
    i=image[index]
    img_one=[]
    lidar_one=[]
    ground_thuth_one=[]
    

    
    img_file = Image.open(image_path+'/'+i)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    img=rgb_png
    
    img_file = Image.open(velodyne_raw_path+  '/'+i[:27]+'velodyne_raw'+i[32:])
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    depth = depth_png.astype(np.float32) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)
    
    
    img_file = Image.open(ground_truth_path+  '/'+i[:27]+'groundtruth_depth'+i[32:])
    ground_truth = np.array(img_file, dtype=int)
    img_file.close()
    ground_truth = ground_truth.astype(np.float) / 256.


    img_one.append(img)
    lidar_one.append(depth[:,:,0])
    ground_thuth_one.append(ground_truth)
    if with_semantic:
        semantic = Image.open('../depth_selection/val_selection_cropped/imagesegmented_images'+'/'+i)
        rgb_png = np.array(semantic, dtype='uint8') # in the range [0,255]
        img_file.close()
        img=rgb_png
        
    return  np.asarray(img_one),np.asarray(lidar_one),np.asarray(ground_thuth_one), img

def read_one_test(index):
    ground_truth_path='../depth_selection/test_depth_completion_anonymous/groundtruth_depth'
    velodyne_raw_path='../depth_selection/test_depth_completion_anonymous/velodyne_raw'
    image_path='../depth_selection/test_depth_completion_anonymous/image'
    image=os.listdir('../depth_selection/test_depth_completion_anonymous/image')
    velodyne_raw=os.listdir('../depth_selection/test_depth_completion_anonymous/velodyne_raw')
    intrinsics=os.listdir('../depth_selection/test_depth_completion_anonymous/intrinsics')
    
    i=image[index]
    img_one=[]
    lidar_one=[]
    ground_thuth_one=[]
    

    
    img_file = Image.open(image_path+'/'+i)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    img=rgb_png
    
    img_file = Image.open(velodyne_raw_path+  '/'+i[:27])
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)
    
    
    
    img_one.append(img)
    lidar_one.append(depth[:,:,0])

    return  np.asarray(img_one).astype(np.float32),np.asarray(lidar_one).astype(np.float32),image[index]







class Data_load_NYU():
    def __init__(self,sample_rate=64):
        self.sample_rate=int(sample_rate)
        self.clip_size_x=0
        self.clip_size_y=0
        self.resize=True

        self.depth_path = '../depth_selection/NYU_depth/train'+'/*/*.h5'
    
        self.filenames = glob.glob(self.depth_path)


        self.num_sample=[i for i in range(len(self.filenames))]
        np.random.shuffle(self.num_sample)
        self.index=0
        self.total_sample=len(self.filenames)
      
    def read_batch(self,batch_size=4):
        i=0
        img_batch=[]
        lidar_batch=[]
        gt_batch=[]

        while (i<(batch_size)):
            i=i+1
            one_sample=h5py.File(self.filenames[self.index], 'r')
            
            #img=rgb_read(img_path[index])
            #print (len(self.num_sample))
            ground_truth=one_sample['depth']
            rgb=one_sample['rgb']
            rgb=np.transpose(rgb, (1, 2, 0))
            if self.resize:
                rgb=resize(rgb,(240,320))
                ground_truth=resize(ground_truth,(240,320))

            height,width=np.shape(ground_truth)
            rgb=rgb[int(self.clip_size_y/2):int(height-self.clip_size_y/2),int(self.clip_size_x/2):int(width-self.clip_size_x/2),:]
            ground_truth=ground_truth[int(self.clip_size_y/2):int(height-self.clip_size_y/2),int(self.clip_size_x/2):int(width-self.clip_size_x/2)]


            
            Mask=np.zeros((int(height-self.clip_size_y),int(width-self.clip_size_x)))
            first_index=np.random.randint(height-12, size=self.sample_rate)
            second_index=np.random.randint(width-16, size=self.sample_rate)
            Mask[first_index+6,second_index+8]=1
            depth=ground_truth*Mask
            depth=np.expand_dims(depth,axis=-1)
            ground_truth=np.expand_dims(ground_truth,axis=-1)

            lidar_batch.append(depth)
            img_batch.append(rgb*255)
            gt_batch.append(ground_truth)
            self.index=self.index+1
        if self.index+batch_size>self.total_sample:
            return [0],[1],[2]
        else:
            return  np.asarray(lidar_batch).astype(np.float32),np.asarray(gt_batch).astype(np.float32),np.asarray(img_batch).astype(np.float32)

def read_one_val_NYU(index,sample_rate=64, clip_size_x=0,clip_size_y=0,if_resize=True):

    depth_path = '../depth_selection/NYU_depth/val'+'/*/*.h5'
    filenames = glob.glob(depth_path)

    img_one=[]
    lidar_one=[]
    gt_one=[]

    one_sample=h5py.File(filenames[index], 'r')
    ground_truth=one_sample['depth']
    rgb=one_sample['rgb']
    rgb=np.transpose(rgb, (1, 2, 0))
    if if_resize:
        rgb=resize(rgb,(240,320))
        ground_truth=resize(ground_truth,(240,320))

    height,width=np.shape(ground_truth)
    rgb=rgb[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2),:]
    ground_truth=ground_truth[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2)]


            
    Mask=np.zeros((int(height-clip_size_y),int(width-clip_size_x)))
    first_index=np.random.randint(height-12, size=sample_rate)
    second_index=np.random.randint(width-16, size=sample_rate)
    Mask[first_index+6,second_index+8]=1
    depth=ground_truth*Mask
            
    lidar_one.append(depth)
    img_one.append(rgb*255)
    gt_one.append(ground_truth)

        
    return  np.asarray(img_one).astype(np.float32),np.asarray(lidar_one).astype(np.float32),np.asarray(gt_one).astype(np.float32)




class KITTI_demo_loader():
    
    def __init__(self):
        self.lidar_sequence=[]
        self.rgb_sequence=[]
        self.demo_KITTI()
        self.index=0

    
    def demo_KITTI(self):
        RGB_image_path=glob.glob('../depth_selection/Demo/KITTI/*rgb/image_02/data/*.png')
        Lidar_image_path=glob.glob('../depth_selection/Demo/KITTI/*lidar/proj_depth/*/image_02/*.png')
        for i in Lidar_image_path:
            temp=i.split('/')[-1]
            for j in RGB_image_path:
                if j.split('/')[-1]==temp:
                    self.lidar_sequence.append(i)
                    self.rgb_sequence.append(j)
                    
    def rgb_read(self,filename):
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
        rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
        rgb_png =np.array(Image.fromarray(rgb_png).resize((1216,352), Image.NEAREST))
        img_file.close()
        return rgb_png
    
    def depth_read(self,filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

        depth = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.

        depth  = np.array(Image.fromarray(depth).resize((1216,352), Image.NEAREST))

        depth = np.expand_dims(depth,-1)
        return depth
    
    def read_one_image(self):
        lidar_ind_path=self.lidar_sequence[self.index]
        rgb_ind_path=self.rgb_sequence[self.index]
        name_ind_path=lidar_ind_path.split('/')
        self.index=self.index+1
        return self.rgb_read(rgb_ind_path),self.depth_read(lidar_ind_path),name_ind_path[-1]
        
class NYU_demo_loader():
    
    def __init__(self):
        self.filenames=[]
        self.demo_NYU()
        self.index=0

    
    def demo_NYU(self):
        depth_path = '../depth_selection/Demo/NYU/dining_room/*.h5'
        self.filenames = glob.glob(depth_path)
        
    def read_one_val_NYU(self,file_name,sample_rate=500, clip_size_x=0,clip_size_y=0,if_resize=True):
        img_one=[]
        lidar_one=[]
        gt_one=[]
  
        one_sample=h5py.File(file_name, 'r')
        ground_truth=one_sample['depth']
        rgb=one_sample['rgb']
        rgb=np.transpose(rgb, (1, 2, 0))
        if if_resize:
            rgb=resize(rgb,(240,320))
            ground_truth=resize(ground_truth,(240,320))

        height,width=np.shape(ground_truth)
        rgb=rgb[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2),:]
        ground_truth=ground_truth[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2)]



        Mask=np.zeros((int(height-clip_size_y),int(width-clip_size_x)))
        first_index=np.random.randint(height-12, size=sample_rate)
        second_index=np.random.randint(width-16, size=sample_rate)
        Mask[first_index+6,second_index+8]=1
        depth=ground_truth*Mask

        lidar_one.append(depth)
        img_one.append(rgb*255)
        gt_one.append(ground_truth)


        return  np.asarray(img_one).astype(np.float32),np.asarray(lidar_one).astype(np.float32)

        
 
    def read_one_image(self):
        lidar_ind_path=self.filenames[self.index]

        self.index=self.index+1
        rgb,lidar=self.read_one_val_NYU(lidar_ind_path)
        return rgb,lidar,lidar_ind_path.split('/')[-1]


class Data_load_NYU_torch():
    def __init__(self,sample_rate=64):
        self.sample_rate=int(sample_rate)
        self.clip_size_x=0
        self.clip_size_y=0
        self.resize=True

        self.depth_path = '../depth_selection/NYU_depth/train'+'/*/*.h5'
    
        self.filenames = glob.glob(self.depth_path)


        self.num_sample=[i for i in range(len(self.filenames))]
        np.random.shuffle(self.num_sample)
        self.index=0
        self.total_sample=len(self.filenames)
      
    def read_batch(self,batch_size=4):
        i=0
        img_batch=[]
        lidar_batch=[]
        gt_batch=[]

        while (i<(batch_size)):
            i=i+1
            one_sample=h5py.File(self.filenames[self.index], 'r')
            
            #img=rgb_read(img_path[index])
            #print (len(self.num_sample))
            ground_truth=one_sample['depth']
            rgb=one_sample['rgb']
            rgb=np.transpose(rgb, (1, 2, 0))
            if self.resize:
                rgb=resize(rgb,(256,320))
                ground_truth=resize(ground_truth,(256,320))

            height,width=np.shape(ground_truth)
            rgb=rgb[int(self.clip_size_y/2):int(height-self.clip_size_y/2),int(self.clip_size_x/2):int(width-self.clip_size_x/2),:]
            ground_truth=ground_truth[int(self.clip_size_y/2):int(height-self.clip_size_y/2),int(self.clip_size_x/2):int(width-self.clip_size_x/2)]


            
            Mask=np.zeros((int(height-self.clip_size_y),int(width-self.clip_size_x)))
            first_index=np.random.randint(height-12, size=self.sample_rate)
            second_index=np.random.randint(width-16, size=self.sample_rate)
            Mask[first_index+6,second_index+8]=1
            depth=ground_truth*Mask
            depth=np.expand_dims(depth,axis=-1)
            ground_truth=np.expand_dims(ground_truth,axis=-1)

            lidar_batch.append(depth)
            img_batch.append(rgb*255)
            gt_batch.append(ground_truth)
            self.index=self.index+1
        if self.index+batch_size>self.total_sample:
            return [0],[1],[2]
        else:
            return  np.asarray(lidar_batch).astype(np.float32),np.asarray(gt_batch).astype(np.float32),np.asarray(img_batch).astype(np.float32)



def read_one_val_NYU_torch(index,sample_rate=64, clip_size_x=0,clip_size_y=0,if_resize=True):

    depth_path = '../depth_selection/NYU_depth/val'+'/*/*.h5'
    filenames = glob.glob(depth_path)

    img_one=[]
    lidar_one=[]
    gt_one=[]

    one_sample=h5py.File(filenames[index], 'r')
    ground_truth=one_sample['depth']
    rgb=one_sample['rgb']
    rgb=np.transpose(rgb, (1, 2, 0))
    if if_resize:
        rgb=resize(rgb,(256,320))
        ground_truth=resize(ground_truth,(256,320))

    height,width=np.shape(ground_truth)
    rgb=rgb[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2),:]
    ground_truth=ground_truth[int(clip_size_y/2):int(height-clip_size_y/2),int(clip_size_x/2):int(width-clip_size_x/2)]


            
    Mask=np.zeros((int(height-clip_size_y),int(width-clip_size_x)))
    first_index=np.random.randint(height-12, size=sample_rate)
    second_index=np.random.randint(width-16, size=sample_rate)
    Mask[first_index+6,second_index+8]=1
    depth=ground_truth*Mask
            
    lidar_one.append(depth)
    img_one.append(rgb*255)
    gt_one.append(ground_truth)

        
    return  np.asarray(img_one).astype(np.float32),np.asarray(lidar_one).astype(np.float32),np.asarray(gt_one).astype(np.float32)
