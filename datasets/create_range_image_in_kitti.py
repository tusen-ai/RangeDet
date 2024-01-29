#keep full resolution range image
import numpy as np
import pickle as pkl
from pdb import set_trace
import os
from kitti_utils import Calibration
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
import collections
import argparse
from queue import Queue
from threading import Thread


def parse_args():
    parser = argparse.ArgumentParser(description='Create range images in KITTI')
    parser.add_argument('--source-dir', help='path to KITTI in MMDet3D format', type=str)
    parser.add_argument('--target-dir', help='path to save the extracted data')
    parser.add_argument('--num-threads', help='path to save the extracted data', type=int, default=10)

    args = parser.parse_args()
    return args

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:
    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    """
    xyz_camera = boxes3d_camera[:, 0:3]
    l, h, w, r = boxes3d_camera[:, 3:4], boxes3d_camera[:, 4:5], boxes3d_camera[:, 5:6], boxes3d_camera[:, 6:7]
    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

def to_xyz0z1(bbox_type7): #[n, r, 7]
    batch_size, num_bbox, _ = bbox_type7.shape
    dtype = bbox_type7.dtype
    
    xy_4pts = np.full((batch_size, num_bbox, 4, 2), 0, dtype = dtype)
    xy_4pts[:,:,0,:] = np.array([[[ 0.5, -0.5]]], dtype = dtype) * bbox_type7[:,:,3:5]
    xy_4pts[:,:,1,:] = np.array([[[-0.5, -0.5]]], dtype = dtype) * bbox_type7[:,:,3:5]
    xy_4pts[:,:,2,:] = np.array([[[-0.5,  0.5]]], dtype = dtype) * bbox_type7[:,:,3:5]
    xy_4pts[:,:,3,:] = np.array([[[ 0.5,  0.5]]], dtype = dtype) * bbox_type7[:,:,3:5]
    cosa = np.cos(bbox_type7[:,:,-1])
    sina = np.sin(bbox_type7[:,:,-1])
    rot_mat = np.stack([cosa, -sina, sina, cosa], axis = -1).reshape(batch_size, num_bbox, 2, 2)
    rot_4pts = np.einsum('nrij,nrmj->nrmi', rot_mat, xy_4pts)
    rot_4pts = rot_4pts + bbox_type7[:,:,None,:2]
    rot_4pts = rot_4pts.reshape(batch_size, num_bbox, 8)
    
    z0 = bbox_type7[:,:,2] - bbox_type7[:,:,5] / 2
    z1 = bbox_type7[:,:,2] + bbox_type7[:,:,5] / 2
    bbox_xyz0z1 = np.concatenate([rot_4pts, z0[:,:,None], z1[:,:,None]], axis = 2)
    return bbox_xyz0z1

def to_8pts(bbox_4pts):
    bbox_4pts = bbox_4pts.astype(np.float32)
    xy = bbox_4pts[:,:8].reshape(-1,4,2)
    z_bot = bbox_4pts[:,8]
    z_bot = np.tile(z_bot[:,None],(1,4))

    z_top = bbox_4pts[:,9]
    z_top = np.tile(z_top[:,None],(1,4))

    xyz_bot = np.concatenate([xy, z_bot[:,:,None]],axis = 2)
    xyz_top = np.concatenate([xy, z_top[:,:,None]],axis = 2)
    bbox_8pts = np.concatenate([xyz_bot,xyz_top], axis = 1)
    return bbox_8pts

def name_to_cls(names):
    cls_mapping = {'Car':1, 'Pedestrian':2, 'Cyclist':4}
    gt_class = []
    for name in names:
        if name in cls_mapping:
            gt_class.append(cls_mapping[name])
        else:
            gt_class.append(-1)
    gt_class = np.array(gt_class)
    return gt_class

def get_pc(target_dir, pc_idx, is_test=False):
    if is_test:
        path = '{}/testing/velodyne/{}.bin'.format(target_dir, pc_idx)
    else:
        path = '{}/training/velodyne/{}.bin'.format(target_dir, pc_idx)
    pc = np.fromfile(path, dtype = np.float32).reshape(-1, 4)
    return pc

def get_gt_bbox(location, dimensions, rotation_y, calib):
    '''
        location: object location x,y,z in camera coordinates (in meters)
        dimensions   3D object dimensions: height, width, length (in meters)
        rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    '''
    gt_bbox_camera = np.concatenate([location, dimensions, rotation_y[:,None]], axis = 1).astype(np.float32)
    gt_bbox_lidar =  boxes3d_kitti_camera_to_lidar(gt_bbox_camera, calib)
    bbox_xyz0z1 = to_xyz0z1(gt_bbox_lidar[None, :, :]).squeeze(0)
    bbox_8pts = to_8pts(bbox_xyz0z1)
    return bbox_8pts



def get_range_image(pc, incl, height):
    incl_deg = incl * 180 / 3.1415
    # print(incl - np.roll(incl, 1))
    xy_norm = np.linalg.norm(pc[:, :2], ord = 2, axis = 1)
    error_list = []
    for i in range(len(incl)):
        h = height[i]
        theta = incl[i]
        error = np.abs(theta - np.arctan2(h - pc[:,2], xy_norm))
        error_list.append(error)
    all_error = np.stack(error_list, axis=-1)
    row_inds = np.argmin(all_error, axis=-1)

    azi = np.arctan2(pc[:,1], pc[:,0])
    width = 2048
    col_inds = width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * width
    col_inds = np.round(col_inds).astype(np.int32)
    col_inds[col_inds == width] = width - 1
    col_inds[col_inds < 0] = 0
    empty_range_image = np.full((64, width, 5), -1, dtype = np.float32)
    point_range = np.linalg.norm(pc[:,:3], axis = 1, ord = 2)

    order = np.argsort(-point_range)
    point_range = point_range[order]
    pc = pc[order]
    row_inds = row_inds[order]
    col_inds = col_inds[order]

    empty_range_image[row_inds, col_inds, :] = np.concatenate([point_range[:,None], pc], axis = 1)

    return empty_range_image




def get_calib(source_dir, idx, is_test):
    # if is_test:
    #     path = '/mnt/truenas/scratch/zhichao.li/Data/KITTI/testing/calib'
    # else:
    #     path = '/mnt/truenas/scratch/zhichao.li/Data/KITTI/training/calib'
    if is_test:
        path = os.path.join(source_dir, 'testing/calib')
    else:
        path = os.path.join(source_dir, 'training/calib')
    calib_file =  os.path.join(path, '{}.txt'.format(idx))
    p = Path(calib_file)
    assert p.exists()
    return Calibration(p)

def crop_range_image(range_image):
    # width = 2083 // 4
    mid = 2083 // 2
    beg = mid - 256
    end = mid + 256
    return range_image[:,beg:end,:]

def process_single_frame(frame, source_dir, target_dir, split, roidb_list):
    pc_idx = frame['point_cloud']['lidar_idx']

    if split != 'test':
        calib = get_calib(source_dir, pc_idx, split=='test')
        annos = frame['annos']
        gt_class = name_to_cls(annos['name'])
        gt_bbox = get_gt_bbox(annos['location'], annos['dimensions'], annos['rotation_y'], calib)
    else:
        gt_class = np.ones(0,dtype=np.float32)
        gt_bbox = np.zeros(0,dtype=np.float32)

    pc = get_pc(source_dir, pc_idx, split=='test')
    pc_url = os.path.join(target_dir, '{}/{}.npz'.format(npz_dirname, pc_idx))
    range_image = get_range_image(pc, incl, height)
    range_image_mask = range_image[..., 0] > -1

    roidb = {
        'gt_class':gt_class,
        'gt_bbox_imu':gt_bbox,
        'pc_url':pc_url
    }

    roidb_list.append(roidb)
    np.savez(
        pc_url,
        range_image=range_image,
        range_image_mask=range_image_mask,
    )

def process_task_worker(frame_queue, source_dir, target_dir, split, roidb_list):
    while True:
        qsize = frame_queue.qsize()
        if  qsize > 0:
            if qsize % 10 == 0:
                print('{} {} frames left'.format(qsize, split))
            frame = frame_queue.get()
        else:
            print("No task left, break down.")
            break
        try:
            process_single_frame(frame, source_dir, target_dir, split, roidb_list)
        except Exception as e:
            print('Error: ', e)
            continue

if __name__ == '__main__':
    # KITTI scanning parameters, obtained from Hough transformation
    height = np.array(
      [0.20966667, 0.2092    , 0.2078    , 0.2078    , 0.2078    ,
       0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
       0.20453333, 0.205     , 0.2036    , 0.20406667, 0.2036    ,
       0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008    ,
       0.2008    , 0.2008    , 0.20033333, 0.1994    , 0.20033333,
       0.19986667, 0.1994    , 0.1994    , 0.19893333, 0.19846667,
       0.19846667, 0.19846667, 0.12566667, 0.1252    , 0.1252    ,
       0.12473333, 0.12473333, 0.1238    , 0.12333333, 0.1238    ,
       0.12286667, 0.1224    , 0.12286667, 0.12146667, 0.12146667,
       0.121     , 0.12053333, 0.12053333, 0.12053333, 0.12006667,
       0.12006667, 0.1196    , 0.11913333, 0.11866667, 0.1182    ,
       0.1182    , 0.1182    , 0.11773333, 0.11726667, 0.11726667,
       0.1168    , 0.11633333, 0.11633333, 0.1154    ])
    zenith = np.array([
        0.03373091,  0.02740409,  0.02276443,  0.01517224,  0.01004049,
        0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
       -0.02609267, -0.032068  , -0.03853542, -0.04451074, -0.05020488,
       -0.0565317 , -0.06180405, -0.06876355, -0.07361411, -0.08008152,
       -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
       -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
       -0.14510716, -0.15213696, -0.1575499 , -0.16711043, -0.17568678,
       -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
       -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
       -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
       -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908  ,
       -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
       -0.40703745, -0.41835542, -0.42777535, -0.43621111
    ]) 
    incl = -zenith

    args = parse_args()
    data_splits = ['training', 'validation', 'test']
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)
    os.makedirs(target_dir)
    num_threads = args.num_threads
    for split in data_splits:

        if split == 'training':
            npz_dirname = 'npz_trainval'
            info_path = os.path.join(source_dir, 'kitti_infos_train.pkl')
        elif split == 'validation':
            npz_dirname = 'npz_trainval'
            info_path = os.path.join(source_dir, 'kitti_infos_val.pkl')
        elif split == 'test':
            npz_dirname = 'npz_test'
            info_path = os.path.join(source_dir, 'kitti_infos_test.pkl')

        npz_dirpath = os.path.join(target_dir, npz_dirname)
        os.makedirs(npz_dirpath)
        # os.makedirs(npz_dirpath, exist_ok=True)
        print(f'Begin processing {split} split, and all created data will be saved under: {target_dir}')
        
        data_set = pkl.load(open(info_path, 'rb'))
        roidb_list = []

        frame_queue = Queue()
        for i, frame in enumerate(data_set):
            frame_queue.put(frame)

        workers = [
            Thread(target=process_task_worker, args=(frame_queue, source_dir, target_dir, split, roidb_list))
            for _ in range(num_threads)]

        for w in workers:
            w.start()

        for w in workers:
            w.join()

        print(f'Got {len(roidb_list)} frame in {split} split.')

        if split == 'training':
            with open(os.path.join(target_dir, 'training.roidb'), 'wb') as fw:
                pkl.dump(roidb_list, fw)
        elif split == 'validation':
            with open(os.path.join(target_dir, 'validation.roidb'), 'wb') as fw:
                pkl.dump(roidb_list, fw)
        elif split == 'test':
            with open(os.path.join(target_dir, 'test.roidb'), 'wb') as fw:
                pkl.dump(roidb_list, fw)

