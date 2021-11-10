import argparse
import pickle as pkl

import numpy as np
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

type_dict = {
    'TYPE_UNKNOWN': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_SIGN': 3,
    'TYPE_CYCLIST': 4,
}

cls_map = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 0
}


def _create_bbox_prediction(det, class_id, frame_name, marco_ts):
    o = metrics_pb2.Object()
    o.context_name = (frame_name)
    o.frame_timestamp_micros = marco_ts
    # This is only needed for 2D detection or tracking tasks.
    # Set it to the camera name the prediction is for.
    # o.camera_name = camera_dict[camera_name]
    # Populating box and score.
    box = label_pb2.Label.Box()
    box.center_x = np.mean(det[[0, 2, 4, 6]])
    box.center_y = np.mean(det[[1, 3, 5, 7]])
    z0 = det[9]
    height = det[10]
    box.center_z = z0 + height / 2
    box.width = np.sqrt((det[2] - det[4]) ** 2 + (det[3] - det[5]) ** 2)
    box.length = np.sqrt((det[2] - det[0]) ** 2 + (det[3] - det[1]) ** 2)
    box.height = height
    box.heading = det[8]
    o.object.box.CopyFrom(box)

    # This must be within [0.0, 1.0]. It is better to filter those boxes with
    # small scores to speed up metrics computation.
    if len(det) == 12:
        o.score = det[11]  # 1/(1 + np.exp(-1 * det[11]))
    # For tracking, this must be set and it must be unique for each tracked
    # sequence.
    o.object.id = ''
    # Use correct type.

    o.object.type = cls_map[class_id]
    return o


def _create_pd_file_example(obj_list, filename):
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()
    for obj in obj_list:
        objects.objects.append(obj)
    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.

    # Write objects to a file.
    f = open(filename, 'wb')
    f.write(objects.SerializeToString())
    f.close()


def main(exp_path, name, bins_path):
    with open(exp_path, "rb") as fr:
        annotation_dict = pkl.load(fr)
        output_dict = pkl.load(fr)
    np.set_printoptions(
        suppress=True,
        formatter={'float_kind': '{:0.2f}'.format})

    pred_list = []
    count = 0
    for rec_id, output in output_dict.items():
        count += 1
        if count % 5000 == 0:
            print(count, '/', len(output_dict))

        frame_name = output['pc_url'].split('/')[-2].replace('segment-', '').replace('_with_camera_labels', '')
        ts = int(output['pc_url'].split('/')[-1][:-4])

        for i in range(1, 6):
            if i in output['det_4pts'].keys():
                for bbox in output['det_4pts'][i]:
                    pred_list.append(_create_bbox_prediction(bbox, i, frame_name, ts))
    _create_pd_file_example(pred_list, '{}/{}.bin'.format(bins_path, name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '-p', help='path to experiments', type=str, default='')
    parser.add_argument('--config', '-c', help='config name', type=str)
    parser.add_argument('--epoch', '-e', help='the epoch number to test', type=str)
    parser.add_argument('--save_bin_path', '-e', help='path to save bins', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    exp_path = '{}/{}/checkpoint_output_dict_{}e.pkl'.format(
        args.exp_path, args.config, args.epoch)
    main(exp_path, args.config, args.save_bin_path)
