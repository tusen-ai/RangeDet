# %%
import argparse
import glob
import os
import pickle as pkl
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

NUM_THREAD = 40


def parse_args():
    parser = argparse.ArgumentParser(description='Extract data from tfrecord and store in npz or pickle format')
    parser.add_argument('--data_path', help='path to tfrecord', type=str)
    parser.add_argument('--save_path', help='path to save the extracted data')
    parser.add_argument('--dataset-split', help='dataset split, e.g. training, validation and testing', type=str)
    parser.add_argument('--save_dir', help='directory to save the extracted data', type=str)

    args = parser.parse_args()
    return args


def makedirs(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        print('make dir', dir_path)
        os.makedirs(dir_path)
    return dir_path


def get_azimuth(extrinsic, height, width):
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(width, 0, -1) - 0.5) / width
    azimuth = (ratios * 2 - 1) * np.pi - az_correction
    return azimuth.astype(np.float32)


def get_pc_cartesian_image(frame,
                           range_images,
                           camera_projections,
                           range_image_top_pose,
                           ri_index=0):
    """
    This function borrow from https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/utils/frame_utils.py#L81

    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    ############################################################
    for c in calibrations:
        if c.name != open_dataset.LaserName.TOP:
            continue
        ############################################################
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        ############################################################
        range_image_mask = range_image_tensor[..., 0] > 0
        beam_inclinations = tf.convert_to_tensor(value=beam_inclinations)
        ############################################################
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(beam_inclinations, axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        ############################################################
        pixel_pose_local = pixel_pose_local.numpy().squeeze()
        frame_pose_local = frame_pose_local.numpy().squeeze()
        azimuth = get_azimuth(extrinsic, 64, 2650)

        range_image_cartesian = range_image_cartesian.numpy()
        range_image_mask = range_image_mask.numpy()
        beam_inclinations = beam_inclinations.numpy()

        info_pack = {'pc_vehicle_frame': range_image_cartesian,
                     'range_image_mask': range_image_mask,
                     'inclination': beam_inclinations,
                     'azimuth': azimuth,
                     'extrinsic': extrinsic,
                     'frame_pose': frame_pose_local}
        return info_pack
        ############################################################


def convert_range_images_to_numpy(range_images):
    top0 = range_images[open_dataset.LaserName.TOP][0]
    # top1 = range_images[open_dataset.LaserName.TOP][1]
    top0_tf = tf.convert_to_tensor(top0.data)
    # top1_tf = tf.convert_to_tensor(top1.data)
    top0_tf = tf.reshape(top0_tf, top0.shape.dims)
    # top1_tf = tf.reshape(top0_tf, top1.shape.dims)
    top0_np = top0_tf.numpy()
    # top1_np = top1_tf.numpy()
    return top0_np


def get_data_from_seg(segment, args):
    tf_dataset = tf.data.TFRecordDataset(segment, compression_type='')

    seg_name = segment.split('/')[-1].split('.')[0]
    makedirs(os.path.join(args.save_path, args.save_dir) + '/{}/'.format(seg_name))

    roidb = []
    for data in tf_dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        data_pack = get_pc_cartesian_image(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0)

        range_image_np = \
            convert_range_images_to_numpy(range_images)
        data_pack['range_image'] = range_image_np

        bboxes = []
        meta_data = []
        points_in_box = []
        label_type = []
        if args.dataset_split != 'testing':
            for label in frame.laser_labels:
                bbox = []
                bbox.append(label.box.center_x)
                bbox.append(label.box.center_y)
                bbox.append(label.box.center_z)
                bbox.append(label.box.length)
                bbox.append(label.box.width)
                bbox.append(label.box.height)
                bbox.append(label.box.heading)
                bboxes.append(bbox)
                meta = []
                meta.append(label.metadata.speed_x)
                meta.append(label.metadata.speed_y)
                meta.append(label.metadata.accel_x)
                meta.append(label.metadata.accel_y)
                meta_data.append(meta)
                if hasattr(label, 'num_lidar_points_in_box'):
                    points_in_box.append(label.num_lidar_points_in_box)
                else:
                    points_in_box.append(-1)
                label_type.append(label.type)
        bboxes = np.array(bboxes)
        if len(bboxes) > 0:
            corners = box_utils.get_upright_3d_box_corners(bboxes).numpy()
        else:
            corners = np.empty((0,))

        save_pc_url = os.path.join(
            args.save_path, args.save_dir
        ) + '/{}/{}.npz'.format(seg_name, frame.timestamp_micros)

        np.savez(save_pc_url,
                 **data_pack,
                 )

        # 3d points in vehicle frame.
        annotation = {
            'pc_url': save_pc_url,
            'gt_class': np.array(label_type),
            'gt_bbox_yaw': bboxes[:, 6] if len(bboxes) > 0 else np.empty((0,)),
            'gt_bbox_csa': bboxes,
            'gt_bbox_imu': corners,
            'meta_data': np.array(meta_data),
            'points_in_box': np.array(points_in_box)
        }

        roidb.append(annotation)

    pkl.dump(roidb, open(args.save_path + '/{}/{}.roidb'.format(args.dataset_split, seg_name), 'wb'))


# %%
def process_task_worker(bag_dict_queue, args):
    while True:
        if bag_dict_queue.qsize() > 0:
            print('{} segments left'.format(bag_dict_queue.qsize()))
            bag_dict = bag_dict_queue.get()
        else:
            print("No task left, break down.")
            break
        try:
            get_data_from_seg(bag_dict, args)
        except:
            print('unknown error')
            continue


if __name__ == '__main__':
    args = parse_args()
    segments = glob.glob(os.path.join(args.data_path, args.dataset_split, '*.tfrecord'))
    # get_data_from_seg(segments[100])
    os.makedirs(os.path.join(args.save_path, args.dataset_split))
    bag_dict_queue = Queue()
    for i, segment in enumerate(segments):
        seg_name = segment.split('/')[-1].split('.')[0]
        if not os.path.exists(args.save_path + '/{}/{}.roidb'.format(args.dataset_split, seg_name)):
            print('put {} in data queue'.format(segment))
            bag_dict_queue.put(segment)

    workers = [
        Thread(target=process_task_worker, args=(bag_dict_queue, args))
        for _ in range(NUM_THREAD)]

    for w in workers:
        # w.daemon = True
        w.start()
