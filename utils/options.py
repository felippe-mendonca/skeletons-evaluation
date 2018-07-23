import os
import json

def load_options(filename='global_options.json'):
  if not os.path.exists(filename):
    raise Exception('Options file \'{}\' doesn\'t exist.'.format(filename))
  with open(filename, 'r') as f:
    return json.load(f)