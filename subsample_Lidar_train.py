
import glob
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import imageio
def get_paths_and_transform():
    
    root_d = os.path.join('./depth_selection/KITTI/Sparse_Lidar')
    
    root_rgb = os.path.join('./depth_selection/KITTI/RGB')
    
    glob_sparse_lidar = "train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png"

    glob_sparse_lidar = os.path.join(root_d,glob_sparse_lidar)
    all_lidar_path_with_new=glob.glob(glob_sparse_lidar)
    all_lidar_path_without_new=[i for i in all_lidar_path_with_new if not (('left' in i) or('right' in i)) ]
    paths_sparse_lidar = sorted(all_lidar_path_without_new)
    
    return paths_sparse_lidar



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

    #depth = np.expand_dims(depth,-1)
    return depth


def read_intrinsic(filename):

    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open(filename, "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[0,2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[1,2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K



def read_extrinsic(filename):
    calib = open(filename, "r")
    lines = calib.readlines()
    R_matrix = lines[1]
    T_matrix = lines[2]

    R_matrix = R_matrix.split(":")[1].split(" ")[1:]
    R_matrix = np.reshape(np.array([float(p) for p in R_matrix]),
                      (3, 3)).astype(np.float32)

    T_matrix = T_matrix.split(":")[1].split(" ")[1:]

    T_matrix = np.reshape(np.array([float(p) for p in T_matrix]),
                      (3, 1)).astype(np.float32)

    extrinsic=np.hstack((R_matrix,T_matrix))
    extra_row=[0,0,0,1]
    extra_row=np.asarray(extra_row)
    extra_row=np.reshape(extra_row,(1,4))
    extrinsic=np.vstack((extrinsic,extra_row))


    return extrinsic
    




def prepare_one_sample(filename):
    # filename is the path of lidar
    path_seg=filename.split('/')

    intrinsic_path=path_seg[0]+'/'+path_seg[1]+'/'+path_seg[2]+'/'+'calib'+'/'+path_seg[5][:10]+'/'+'calib_cam_to_cam.txt'
    extrinsic_path=path_seg[0]+'/'+path_seg[1]+'/'+path_seg[2]+'/'+'calib'+'/'+path_seg[5][:10]+'/'+'calib_velo_to_cam.txt'

    sparse_depth=depth_read(filename)
    intrinsic=read_intrinsic(intrinsic_path)
    extrinsic=read_extrinsic(extrinsic_path)

    return sparse_depth,intrinsic,extrinsic


def image_coor_tensor_function():
    x_axis=[i for i in range(1216)]
    x_axis=np.reshape(x_axis,[1216,1])
    x_image=np.tile(x_axis, 352)
    x_image=np.transpose(x_image)
    y_axis=[i for i in range(352)]
    y_axis=np.reshape(y_axis,[352,1])
    y_image=np.tile(y_axis, 1216)
    z_image=np.ones((352,1216))
    image_coor_tensor=[x_image,y_image,z_image]
    image_coor_tensor=np.asarray(image_coor_tensor).astype(np.float32)
    image_coor_tensor=tf.transpose(image_coor_tensor,[1,0,2])
    return image_coor_tensor
    

def get_all_points(lidar,intrinsic,extrinsic,image_coor_tensor):

    lidar_32=np.squeeze(lidar).astype(np.float32)


    intrinsic=np.reshape(intrinsic,[3,3]).astype(np.float32)
    intrinsic_inverse=np.linalg.inv(intrinsic)
    points_homo=tf.linalg.matmul(intrinsic_inverse,image_coor_tensor)

    lidar_32=tf.reshape(lidar_32,[352,1,1216])
    points_homo=points_homo*lidar_32
    extra_image=np.ones((352,1216)).astype(np.float32)
    extra_image=tf.reshape(extra_image,[352,1,1216])
    points_homo=tf.concat([points_homo,extra_image],axis=1)

    #extrinsic_v_2_c=[[0.007,-1,0,0],[0.0148,0,-1,-0.076],[1,0,0.0148,-0.271],[0,0,0,1]]
    extrinsic_v_2_c=extrinsic
    extrinsic_v_2_c=np.reshape(extrinsic_v_2_c,[4,4]).astype(np.float32)
    extrinsic_c_2_v=np.linalg.inv(extrinsic_v_2_c)
    points_lidar=np.matmul(extrinsic_c_2_v,points_homo)


    mask=np.squeeze(lidar)>0.1
    total_points=[points_lidar[:,0,:][mask],points_lidar[:,1,:][mask],points_lidar[:,2,:][mask]]
    total_points=np.asarray(total_points)
    total_points=np.transpose(total_points)
    
    return total_points


def calculate_angle(points):
    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    distance = np.linalg.norm(points, 2, axis=1)
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / distance)
    num_points=np.shape(pitch)[0]
    pitch=np.reshape(pitch,(num_points,1))

    return np.hstack((points,pitch))
    
def sample(all_points_with_angle,keep_ratio=0.25):
    pitch=all_points_with_angle[:,3]
    max_pitch=np.max(pitch)
    min_pitch=np.min(pitch)
    angle_interval=(max_pitch-min_pitch)/64.0
    angle_label=np.ceil((pitch-min_pitch)/angle_interval)
    keep_points=angle_label%(1.0/keep_ratio)==0
    return all_points_with_angle[keep_points][:,:3]


def map_points_on_image(left_points,intrinsic,extrinsic):
    depth_map=np.zeros((352,1216))
    num_left_points,_=np.shape(left_points)
    extra_one=np.ones((num_left_points,1))
    left_points=np.hstack((left_points,extra_one))
    image_homo=np.dot(extrinsic,np.transpose(left_points))
    image_homo=image_homo[:3,:]
    image_homo=np.dot(intrinsic,image_homo)
    z=image_homo[2,:]
    u=np.round(image_homo[0,:]/z).astype(np.int)
    v=np.round(image_homo[1,:]/z).astype(np.int)

    depth_map[v,u]=z
    return depth_map
    





path=get_paths_and_transform()



if not os.path.exists('./depth_selection/KITTI/Sparse_Lidar_32'):
    os.mkdir('./depth_selection/KITTI/Sparse_Lidar_32')



for i in path:
    name_list=i.split('/')
    save_path='./depth_selection/KITTI/Sparse_Lidar_32'
    for j in range(len(name_list)):
        if j >3 :
            save_path=save_path+'/'+name_list[j]
            if j<(len(name_list)-1) and (not os.path.exists(save_path)):
                os.mkdir(save_path)


    sparse_depth,intrinsic,extrinsic=prepare_one_sample(i)
    image_coor_tensor=image_coor_tensor_function()
    all_points=get_all_points(sparse_depth,intrinsic,extrinsic,image_coor_tensor)
    all_points_with_angle=calculate_angle(all_points)
    left_points=sample(all_points_with_angle,keep_ratio=0.5)
    new_depth_map=map_points_on_image(left_points,intrinsic,extrinsic)
    new_depth_map=new_depth_map*256.0
    new_depth_map = new_depth_map.astype(np.uint16)
            
    im=Image.fromarray(new_depth_map)
           
    imageio.imwrite(save_path,im) 




if not os.path.exists('./depth_selection/KITTI/Sparse_Lidar_16'):
    os.mkdir('./depth_selection/KITTI/Sparse_Lidar_16')



for i in path:
    name_list=i.split('/')
    save_path='./depth_selection/KITTI/Sparse_Lidar_16'
    for j in range(len(name_list)):
        if j >3 :
            save_path=save_path+'/'+name_list[j]
            if j<(len(name_list)-1) and (not os.path.exists(save_path)):
                os.mkdir(save_path)


    sparse_depth,intrinsic,extrinsic=prepare_one_sample(i)
    image_coor_tensor=image_coor_tensor_function()
    all_points=get_all_points(sparse_depth,intrinsic,extrinsic,image_coor_tensor)
    all_points_with_angle=calculate_angle(all_points)
    left_points=sample(all_points_with_angle,keep_ratio=0.25)
    new_depth_map=map_points_on_image(left_points,intrinsic,extrinsic)
    new_depth_map=new_depth_map*256.0
    new_depth_map = new_depth_map.astype(np.uint16)
            
    im=Image.fromarray(new_depth_map)
           
    imageio.imwrite(save_path,im) 



