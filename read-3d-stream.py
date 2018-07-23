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
source = 'dataset' # can be either 'detector' or 'dataset'

for dataset in datasets:
  dataset_folder = os.path.join(options['data_folder'], dataset)
  
  # read 3D information
  filename = os.path.join(dataset_folder, '{}_3d_{}'.format(model, source))
  reader = ProtobufReader(filename)
  while True:
    sequence_id = reader.next(Int64Value())
    if not sequence_id:
      break
    objs = reader.next(ObjectAnnotations())
    if not objs:
      break
    n_skeletons = len(objs.objects)
    log.info("[{}] Sequence {} with {} skeletons", dataset, sequence_id, n_skeletons)