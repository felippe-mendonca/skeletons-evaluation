from __future__ import print_function

import os
import json
from is_msgs.image_pb2 import ObjectAnnotations
from google.protobuf.wrappers_pb2 import Int64Value
from utils.io import ProtobufReader
from utils.options import load_options
from is_wire.core import Logger

log = Logger()
options = load_options()

with open('panoptic_datasets.json', 'r') as f:
  panoptic_data = json.load(f)
datasets = panoptic_data['datasets']
model = panoptic_data['model']
source = 'detector' # can be either 'detector' or 'dataset'

for dataset in datasets:
  dataset_folder = os.path.join(options['data_folder'], dataset)
  
  # read 3D information
  filename = os.path.join(dataset_folder, '{}_3d_dataset'.format(model))
  reader = ProtobufReader(filename)
  filename = os.path.join(dataset_folder, '{}_3d_{}_grouped'.format(model, source))
  detections_reader = ProtobufReader(filename)
  
  while True:
    sid = reader.next(Int64Value())
    sid_detections = detections_reader.next(Int64Value())
    if not sid or not sid_detections:
      break
    objs = reader.next(ObjectAnnotations())
    objs_detected = detections_reader.next(ObjectAnnotations())
    
    log.info("{} -> {}", sid.value, sid_detections.value)