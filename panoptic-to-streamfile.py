from __future__ import print_function

import os
import json
import numpy as np
from is_msgs.image_pb2 import ObjectAnnotations
from google.protobuf.wrappers_pb2 import Int64Value
from google.protobuf.json_format import MessageToJson
from is_wire.core import Logger
from utils.io import ProtobufWriter
from utils.options import load_options
from utils.vision import to_camera, resolution_validation
from utils.panoptic import get_model_folder, get_detection_id, get_joints_key
from utils.panoptic import load_calibrations, joints_to_object_annotation
from utils.panoptic import DETECTION_FILE

log = Logger()
options = load_options()

with open('panoptic_datasets.json', 'r') as f:
  panoptic_data = json.load(f)
datasets = panoptic_data['datasets']
model = panoptic_data['model']
cameras = panoptic_data['cameras']
referencial = panoptic_data['referencial']

model_folder = get_model_folder(model)
joints_key = get_joints_key(model)

for dataset in datasets:
  dataset_folder = os.path.join(options['data_folder'], dataset)
  calibs = load_calibrations(dataset_folder, referencial=referencial)
  
  calibs_folder = os.path.join(dataset_folder, 'calibrations')
  if not os.path.exists(calibs_folder):
    os.makedirs(calibs_folder)
  for camera, calib in calibs.items():
    filename = os.path.join(calibs_folder, '{}.json'.format(camera))
    with open(filename, 'w') as f:
      f.write(MessageToJson(calib))
    log.info("[Calibrations][{}] Camera \'{}\' saved", dataset, camera)

  detections_folder = os.path.join(dataset_folder, model_folder)

  def check_file(f, folder=''):
    f = os.path.join(folder, f)
    return os.path.isfile(f) and f.endswith('.json')
  
  detections_files = [f for f in os.listdir(detections_folder) if check_file(f, detections_folder)]  
  detections_ids = list(map(lambda f: get_detection_id(f), detections_files))
  detections_ids.sort()

  output_file = os.path.join(dataset_folder, '{}_2d_dataset'.format(model))
  wirter_2D = ProtobufWriter(output_file)
  output_file = os.path.join(dataset_folder, '{}_3d_dataset'.format(model))
  wirter_3D = ProtobufWriter(output_file)

  log.info("[{}] {} sequences", dataset, len(detections_ids))
  for detection_id in detections_ids:
    detections_file = os.path.join(detections_folder, DETECTION_FILE.format(detection_id))
    with open(detections_file, 'r') as f:
      detections = json.load(f)
    
    sks_3D = ObjectAnnotations()
    sks_3D.frame_id = referencial
    
    sks_2D = {}
    for camera in cameras:
      sks_2D[camera] = ObjectAnnotations()
      sks_2D[camera].resolution.width = calibs[camera].resolution.width
      sks_2D[camera].resolution.height = calibs[camera].resolution.height
    
    for body in detections['bodies']:
      joints_3D = np.array(body[joints_key]).reshape((-1,4)).transpose()[0:3,:]
      sk_3D, invalid_joints = joints_to_object_annotation(joints_3D, model)
      sk_3D.id = body['id'] # on sk_3D, ObjectAnnotation::id stores person id from dataset
      sks_3D.objects.extend([sk_3D])
      for camera in cameras:
        joints_2D = to_camera(joints_3D, calibs[camera], referencial=referencial)
        resolution_valid_joints = resolution_validation(joints_2D, calibs[camera])
        valid_joints = np.logical_and(resolution_valid_joints, np.logical_not(invalid_joints))
        sk_2D, _ = joints_to_object_annotation(joints_2D, model, valid_joints)
        if len(sk_2D.keypoints) > 0:
          sk_2D.id = camera # on sk_2D, ObjectAnnotation::id stores camera id
          sks_2D[camera].objects.extend([sk_2D])

    sequence_id = Int64Value()
    sequence_id.value = detection_id

    wirter_3D.insert(sequence_id)
    wirter_3D.insert(sks_3D)
    
    wirter_2D.insert(sequence_id)
    for camera in cameras:
      wirter_2D.insert(sks_2D[camera])