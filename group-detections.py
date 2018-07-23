from __future__ import print_function

import os
import json
from is_msgs.image_pb2 import ObjectAnnotations
from google.protobuf.wrappers_pb2 import Int64Value
from utils.io import ProtobufReader, ProtobufWriter
from utils.options import load_options
from is_wire.core import Logger

def get_dataset_range(basedir, dataset, cameras, model):
  filename = os.path.join(basedir, dataset, '{}_2d_dataset'.format(model))
  reader = ProtobufReader(filename)
  begin, end, first_read = 0, 0, True
  while True:
      sequence_id = reader.next(Int64Value())
      for camera in cameras:
        _ = reader.next(ObjectAnnotations())
      if not sequence_id:
          break
      if first_read:
          begin = sequence_id.value
          first_read = False
      end = sequence_id.value
  return begin, end

log = Logger()
options = load_options()

with open('panoptic_datasets.json', 'r') as f:
  panoptic_data = json.load(f)
datasets = panoptic_data['datasets']
model = panoptic_data['model']
cameras = panoptic_data['cameras']
get_range = True

for dataset in datasets:
  readers = {}
  for camera in cameras:
    filename = os.path.join(options['data_folder'], dataset, '{}_2d_detector_{}'.format(model, camera))
    readers[camera] = ProtobufReader(filename)

  filename = os.path.join(options['data_folder'], dataset, '{}_2d_detector'.format(model))
  writer = ProtobufWriter(filename)

  begin, end = get_dataset_range(options['data_folder'], dataset, cameras, model) if get_range else (-1, -1)
  
  log.info("[Starting][{}]{} -> {}", dataset, begin, end)
  reading = True
  while reading:
    detections = {}
    sequence_id = Int64Value()
    write = False
    for camera in cameras:
      sequence_id = readers[camera].next(Int64Value())
      if not sequence_id:
        reading = False
        break
      objs = readers[camera].next(ObjectAnnotations())

      if (sequence_id.value >= begin and sequence_id.value <= end) or not get_range:
        write = True
        detections[camera] = objs
        log.info('[{}][{}][{}][{} skeletons]', dataset, camera, sequence_id.value, len(detections[camera].objects))
      elif sequence_id.value > end:
        reading = False
        break
      
    if write:
      writer.insert(sequence_id)
      for camera in cameras:
        writer.insert(detections[camera])
  
  writer.close()
  log.info("[Done][{}]", dataset)