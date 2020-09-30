# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate synthetic data.

Run using the command `python generator_script.py {dataset_name}`. 
The dataset name argument is compulsory.

This file uses dataset_generator.py.

The dataset will be generated at {dataset_dir}/{dataset_name}/
using the config `config`.

This file uses the config to generate a smaller version of the object 
size dataset used in the Robustness Study paper
https://arxiv.org/abs/2007.08558 .

The dataset in the paper includes more object sizes
(object size is written as 'area' in the config):
`areas = list(np.arange(0.01, 1.01, 0.01))`.

The configs for the rotation angle and location datasets are included here:

Config for rotation dataset:
```
angles = list(np.arange(1, 361, 20))
config = {
    'coord': [(0.5, 0.5)],
    'area': [0.2, 0.5, 0.8, 1.0],
    'rotation': angles,
    'bg_resolution': [(512, 512)],
}
dataset_name = 'rotation'
```

Config for location dataset:
```
delta = 0.05
steps = np.arange(0, 1 + delta, delta)
coords = list(itertools.product(steps, steps))

config = {
    'coord': coords,
    'area': [0.2],
    'rotation': [0],
    'bg_resolution': [(500, 500)],
}
dataset_name = 'location'
```
Note that in the paper we generate location datasets with parameter
`min_pct_inside_image` equal to 0.0, 0.5 and 0.75 (instead of the default
0.95 used for the area and rotation datasets).
"""

from os import path

import dataset_generator as synthetic
import numpy as np
import tensorflow.io.gfile as gfile
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='Process dataset location parameters.')
parser.add_argument(
    '--new_dataset_parent_dir',
    default='.',
    help='Directory to create dataset dir.')
parser.add_argument('dataset_name', default='area',
                    help='Name of dataset dir. Entire path is os.path.join('
                    'args.new_dataset_parent_dir, args.dataset_name, "").')
args = parser.parse_args()


def main(args):
  # The config below is the config for the area dataset used in the
  # robustness study paper. The configs for the rotation and location datasets
  # are included above.

  areas = list(np.arange(0.1, 0.9, 0.1))
  areas = [round(x, 2) for x in areas]

  config = {
      'coord': [(0.5, 0.5)],
      'area': areas,
      'rotation': [0],
      'bg_resolution': [(512, 512)],
  }

  new_dataset_dir = path.join(args.new_dataset_parent_dir, args.dataset_name,
                              '')
  if not gfile.exists(new_dataset_dir):
    gfile.makedirs(new_dataset_dir)

  dataset = synthetic.Dataset(
      config=config,
      foregrounds_dir='foreground_samples/',
      backgrounds_dir='background_samples/',
      new_dataset_dir=new_dataset_dir,
      num_bgs_per_fg_instance=2,
      min_pct_inside_image=0.95)

  dataset.generate_dataset()


if __name__ == '__main__':
  main(args)

