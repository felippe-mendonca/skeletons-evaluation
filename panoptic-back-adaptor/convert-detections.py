from __future__ import print_function
import sys
sys.path.insert(0, "../")

import os
import json
from skeletons_pb2 import Skeletons, SkeletonPartType
from is_msgs.image_pb2 import ObjectAnnotations, HumanKeypoints
from google.protobuf.wrappers_pb2 import Int64Value
from is_wire.core import Logger
from utils.options import load_options
from utils.io import ProtobufReader, ProtobufWriter

options = load_options('../global_options.json')

log = Logger()
with open('../panoptic_datasets.json', 'r') as f:
  panoptic_data = json.load(f)
datasets = panoptic_data['datasets']
model = panoptic_data['model']
cameras = panoptic_data['cameras']

for dataset in datasets:
  dataset_folder = os.path.join(options['data_folder'], dataset)
  for camera in cameras:
    log.info("[Starting][{}] camera \'{}\'", dataset, camera)
    filename = os.path.join(dataset_folder, '{}_pose_2d_detected_00_{:02d}'.format(model, camera))
    reader = ProtobufReader(filename)
    filename = os.path.join(dataset_folder, '{}_2d_detector_{}'.format(model, camera))
    writer = ProtobufWriter(filename)

    sid = 0
    while True:
      sks = reader.next(Skeletons())
      if not sks:
        log.info("[Done][{}] camera \'{}\' with {} sequences", dataset, camera, sid + 1)
        break

      objs = ObjectAnnotations()
      # panoptic dataset HD cameras resolution
      objs.resolution.width = 1920
      objs.resolution.height = 1080
      for sk in sks.skeletons:
        obj = objs.objects.add()
        for part in sk.parts:
          type_str = SkeletonPartType.Name(part.type)
          if type_str == 'UNKNOWN' or type_str == 'BACKGROUND':
            continue
          keypoint = obj.keypoints.add()
          keypoint.id = HumanKeypoints.Value(type_str)
          keypoint.score = part.score
          keypoint.position.x = part.x
          keypoint.position.y = part.y

      sequence_id = Int64Value()
      sequence_id.value = sid
      writer.insert(sequence_id)
      writer.insert(objs)
      sid += 1
  