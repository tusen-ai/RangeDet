import argparse
import pickle as pkl

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

type_dict = {
    'TYPE_UNKNOWN': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_SIGN': 3,
    'TYPE_CYCLIST': 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', '-p', help='Path to store experiments', type=str)
    parser.add_argument('--config_name', '-c', help='Name of the config to be tested', type=str)
    parser.add_argument('--epoch', '-e', help='Number of epoch to be tested', type=str)
    parser.add_argument('--save_bin_dir', '-s', help='Path to save bins', type=str)
    args = parser.parse_args()
    return args


def _create_bbox_prediction(bbox3d_xyzlwhyaws, pred_type_id, frame_name, marco_ts):
    """Creates a prediction objects file."""
    o = metrics_pb2.Object()
    # The following 3 fields are used to uniquely identify a frame a prediction
    # is predicted at. Make sure you set them to values exactly the same as what
    # we provided in the raw data. Otherwise your prediction is considered as a
    # false negative.
    o.context_name = (frame_name)
    # The frame timestamp for the prediction. See Frame::timestamp_micros in
    # dataset.proto.
    o.frame_timestamp_micros = marco_ts
    # This is only needed for 2D detection or tracking tasks.
    # Set it to the camera name the prediction is for.
    # o.camera_name = dataset_pb2.CameraName.FRONT

    # Populating box and score.
    box = label_pb2.Label.Box()
    box.center_x = bbox3d_xyzlwhyaws[0]
    box.center_y = bbox3d_xyzlwhyaws[1]
    box.center_z = bbox3d_xyzlwhyaws[2]
    box.length = bbox3d_xyzlwhyaws[3]
    box.width = bbox3d_xyzlwhyaws[4]
    box.height = bbox3d_xyzlwhyaws[5]
    box.heading = bbox3d_xyzlwhyaws[6]
    o.object.box.CopyFrom(box)
    # This must be within [0.0, 1.0]. It is better to filter those boxes with
    # small scores to speed up metrics computation.
    if len(bbox3d_xyzlwhyaws) == 8:
        o.score = bbox3d_xyzlwhyaws[7]
    # For tracking, this must be set and it must be unique for each tracked
    # sequence.
    o.object.id = ''
    # Use correct type.
    o.object.type = pred_type_id
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


def main(pred_boxes_path, config_name, save_bin_dir):
    print(f'Load the prediction boxes from this address: {pred_boxes_path}')
    with open(pred_boxes_path, "rb") as fr:
        annotation_dict = pkl.load(fr)
        output_dict = pkl.load(fr)

    pred_bbox3d_list = []
    for count, (rec_id, output) in enumerate(output_dict.items()):
        if count % 1000 == 0:
            print(f'{count} frames have been processed, a total of {len(output_dict)} frames')
        if len(output) == 0:
            continue

        for pred_type, pred_bboxes3d in output['det_xyzlwhyaws'].items():
            for pred_bbox3d in pred_bboxes3d:
                pred_bbox3d_list.append(
                    _create_bbox_prediction(
                        pred_bbox3d,
                        type_dict[pred_type],
                        output['meta_info']['name'],
                        output['meta_info']['timestamp_micros']))

    _create_pd_file_example(
        pred_bbox3d_list, '{}/{}.bin'.format(save_bin_dir, config_name))


if __name__ == '__main__':
    args = parse_args()
    pred_boxes_path = '{}/{}/checkpoint_output_dict_{}e.pkl'.format(
        args.work_dir, args.config_name, args.epoch)
    main(pred_boxes_path, args.config_name, args.save_bin_dir)
