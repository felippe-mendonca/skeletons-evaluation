from __future__ import print_function
import sys
sys.path.insert(0, "../")

import os
import json
import numpy as np
from functools import reduce
from is_msgs.image_pb2 import ObjectAnnotations
from google.protobuf.wrappers_pb2 import Int64Value
from utils.io import ProtobufReader
from utils.options import load_options
from is_wire.core import Logger
import matplotlib as mpl
import matplotlib.pyplot as plt

log = Logger()
options = load_options('../global_options.json')

if len(sys.argv) != 2:
    log.critical("Enter an options file. python3 make-heatmap.py <OPTIONS_FILE>")

with open(sys.argv[1], 'r') as f:
    hm_options = json.load(f)

dataset = hm_options['dataset']
dataset_folder = os.path.join(options['data_folder'], dataset)
model = hm_options['model']
source = hm_options['source']
filename = os.path.join(dataset_folder, '{}_3d_{}_grouped'.format(model, source))
reader = ProtobufReader(filename)


def get_coordinate(sk, axis, mean=False):
    def reducer(p): return getattr(getattr(p, 'position'), axis)
    coordinates = [reducer(kp) for kp in sk.keypoints]
    return [np.mean(np.array(coordinates))] if mean else coordinates

x, y = [], []
x_axis, y_axis = hm_options['axis_lookup']['x'], hm_options['axis_lookup']['y']
avg_coordinates = hm_options['average_coordinates']
log.info('[Reading detections][{}]', dataset)
while True:
    sequence_id = reader.next(Int64Value())
    if not sequence_id:
        break
    sks = reader.next(ObjectAnnotations())
    if not sks:
        break

    for sk in sks.objects:
        x.extend(get_coordinate(sk, x_axis, avg_coordinates))
        y.extend(get_coordinate(sk, y_axis, avg_coordinates))

log.info('[Reading detections][DONE]')
log.info('[X] [{:.2f}, {:.2f}]', min(x), max(x))
log.info('[Y] [{:.2f}, {:.2f}]', min(y), max(y))

x_min, x_max = hm_options['histogram_bins']['xmin'], hm_options['histogram_bins']['xmax']
y_min, y_max = hm_options['histogram_bins']['ymin'], hm_options['histogram_bins']['ymax']
step = hm_options['histogram_bins']['step']
x_bins, y_bins = np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
x, y = np.array(x), np.array(y)
H, x_edges, y_edges = np.histogram2d(x, y, bins=(x_bins, y_bins))
H = np.log10(H.T + 1.0) if hm_options['log_scale'] else H.T

figsize = (hm_options['figsize']['width'], hm_options['figsize']['height'])
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, aspect='equal', xlim=x_edges[[0, -1]], ylim=y_edges[[0, -1]])
im = mpl.image.NonUniformImage(ax, interpolation='bilinear', cmap='jet')
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
z_centers = (y_edges[:-1] + y_edges[1:]) / 2
im.set_data(x_centers, z_centers, H)
ax.images.append(im)

plt.show()
