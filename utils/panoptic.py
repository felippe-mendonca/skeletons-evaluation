import re
import os
import json
import numpy as np
from is_msgs.image_pb2 import ObjectAnnotation, HumanKeypoints
from is_msgs.camera_pb2 import CameraCalibration
from .numpy import to_tensor

REGEX_DETECTION_ID = re.compile(r'body3DScene_([0-9]+).json')
DETECTION_FILE = 'body3DScene_{:08d}.json'

'''
    http://domedb.perception.cs.cmu.edu/tools.html

    Neck, HeadTop, BodyCenter, lShoulder,lElbow, lWrist, lHip,
    lKnee, lAnkle, rShoulder, rElbow, rWrist, rHip, rKnee, rAnkle
'''
MPII_PB_PARTS = [
    HumanKeypoints.Value('NECK'),           \
    HumanKeypoints.Value('HEAD'),           \
    HumanKeypoints.Value('CHEST'),          \
    HumanKeypoints.Value('LEFT_SHOULDER'),  \
    HumanKeypoints.Value('LEFT_ELBOW'),     \
    HumanKeypoints.Value('LEFT_WRIST'),     \
    HumanKeypoints.Value('LEFT_HIP'),       \
    HumanKeypoints.Value('LEFT_KNEE'),      \
    HumanKeypoints.Value('LEFT_ANKLE'),     \
    HumanKeypoints.Value('RIGHT_SHOULDER'), \
    HumanKeypoints.Value('RIGHT_ELBOW'),    \
    HumanKeypoints.Value('RIGHT_WRIST'),    \
    HumanKeypoints.Value('RIGHT_HIP'),      \
    HumanKeypoints.Value('RIGHT_KNEE'),     \
    HumanKeypoints.Value('RIGHT_ANKLE')
]

# HumanKeypoints.Value('UNKNOWN_HUMAN_KEYPOINT') represents the Background
COCO_PB_PARTS = [
    HumanKeypoints.Value('NECK'),                   \
    HumanKeypoints.Value('NOSE'),                   \
    HumanKeypoints.Value('UNKNOWN_HUMAN_KEYPOINT'), \
    HumanKeypoints.Value('LEFT_SHOULDER'),          \
    HumanKeypoints.Value('LEFT_ELBOW'),             \
    HumanKeypoints.Value('LEFT_WRIST'),             \
    HumanKeypoints.Value('LEFT_HIP'),               \
    HumanKeypoints.Value('LEFT_KNEE'),              \
    HumanKeypoints.Value('LEFT_ANKLE'),             \
    HumanKeypoints.Value('RIGHT_SHOULDER'),         \
    HumanKeypoints.Value('RIGHT_ELBOW'),            \
    HumanKeypoints.Value('RIGHT_WRIST'),            \
    HumanKeypoints.Value('RIGHT_HIP'),              \
    HumanKeypoints.Value('RIGHT_KNEE'),             \
    HumanKeypoints.Value('RIGHT_ANKLE'),            \
    HumanKeypoints.Value('LEFT_EYE'),               \
    HumanKeypoints.Value('LEFT_EAR'),               \
    HumanKeypoints.Value('RIGHT_EYE'),              \
    HumanKeypoints.Value('RIGHT_EAR')
]


def __raise_wrong_model__():
  raise Exception('Invalid model. Can be \'mpii\' (15 joints) or \'coco\' (19 joints)')


def get_model_folder(model):
  model = model.lower()
  if model == 'mpii':
    return 'hdPose3d_stage1'
  elif model == 'coco':
    return 'hdPose3d_stage1_coco19'
  else:
    __raise_wrong_model__()


def get_joints_key(model):
  model = model.lower()
  if model == 'mpii':
    return 'joints15'
  elif model == 'coco':
    return 'joints19'
  else:
    __raise_wrong_model__()


def get_detection_id(filename):
  return int(REGEX_DETECTION_ID.sub(r'\1', filename))


def joints_to_object_annotation(joints, model, valid_joints=None):
    ob = ObjectAnnotation()
    if joints.shape[0] < 2 or joints.shape[0] > 3:
        return ob
    is_3d = joints.shape[0] == 3
    model = model.lower()
    if model == 'coco':
        n_joints = 19
        PB_PARTS = COCO_PB_PARTS
    elif model == 'mpii':
        n_joints = 15
        PB_PARTS = MPII_PB_PARTS
    else:
      __raise_wrong_model__()
    
    if joints.shape[1] != n_joints:
        raise Exception('Invalid number of joints for this model.')
    
    invalid_joints = np.zeros(n_joints, dtype=bool)
    for c in range(n_joints):
        if isinstance(valid_joints, np.ndarray) and not valid_joints[c]:
          invalid_joints[c] = True
          continue
        human_keypoint = PB_PARTS[c]
        if human_keypoint == HumanKeypoints.Value('UNKNOWN_HUMAN_KEYPOINT'):
          invalid_joints[c] = True
          continue
        x, y, z = joints[0, c], joints[1, c], joints[2, c] if is_3d else 0.0
        if all([x == 0.0, y == 0.0, z == 0.0]):
          invalid_joints[c] = True
          continue
        keypoint = ob.keypoints.add()
        keypoint.position.x = x
        keypoint.position.y = y
        if is_3d:
          keypoint.position.z = z
        keypoint.id = human_keypoint

    return ob, invalid_joints


def load_calibrations(dataset_folder, referencial=9999, cameras=None):
    calibration_file = os.path.join(dataset_folder, 'calibration.json')
    if not os.path.exists(calibration_file):
        raise Exception('Calibration file \'{}\' not found.'.format(calibration_file))
    
    with open(calibration_file, 'r') as f:
        calibrations = json.load(f)['cameras']
    
    calibrations = list(filter(lambda d: d['type']=='hd', calibrations))
    calibrations_pb = {}
    for calibration in calibrations:
        calib_pb = CameraCalibration()
        camera = int(calibration['name'].split('_')[1])
        calib_pb.id = camera
        res = calibration['resolution']
        calib_pb.resolution.width=res[0]
        calib_pb.resolution.height=res[1]
        calib_pb.intrinsic.CopyFrom(to_tensor(np.array(calibration['K'])))
        distortion = np.array(calibration['distCoef'])[:,np.newaxis]
        calib_pb.distortion.CopyFrom(to_tensor(distortion))
        extrinsic = calib_pb.extrinsic.add()
        R = np.array(calibration['R'])
        t = np.array(calibration['t'])
        RT = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        setattr(extrinsic, 'from', referencial)
        setattr(extrinsic, 'to', camera)
        extrinsic.tf.CopyFrom(to_tensor(RT))
        if cameras and camera not in cameras:
            continue
        calibrations_pb[camera] = calib_pb

    return calibrations_pb