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

"""Tests for Dataset in `dataset_generator.py`.

Tests methods that are in the Dataset class.
`generator_utils_test.py` tests the methods that are not in the Dataset class.

Run tests using the command `python -m unittest generator_test.py`.
"""

import csv
import os
import unittest

import dataset_generator
import numpy as np
import PIL
import time


class DatasetTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    root_dir = './'
    current_time = int(time.time())
    new_dataset_dir = f'/tmp/test_dir/{current_time}/'
    fg_dir = root_dir + 'foreground_samples/'
    bg_dir = root_dir + 'background_samples/'
    test_config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'area': [0.3, 0.6],
        'rotation': [0],
        'bg_resolution': [(1000, 1000)]
    }

    self.ds = dataset_generator.Dataset(
        test_config,
        new_dataset_dir=new_dataset_dir,
        foregrounds_dir=fg_dir,
        backgrounds_dir=bg_dir,
        num_bgs_per_fg_instance=2)
    self.test_fg = self.ds.fgs_dict['bagel'][0]
    test_bg = dataset_generator.load_image('background_samples/beach2.jpg')
    self.test_bg = self.ds._preprocess_background(test_bg)

    # Parameters of `foreground_samples`, `background_samples` contents.
    self.num_test_fgs = 9
    self.num_test_fg_classes = 9
    self.num_test_bgs = 8

  def test_foreground(self):
    self.assertEqual(len(np.array(self.test_fg)), 683)

  def test_background(self):
    self.assertEqual(len(np.array(self.test_bg)), 1000)

  def test_invalid_fg_bg_dirs(self):
    root_dir = '/temp/'
    new_dataset_dir = '/tmp/test_dir/'
    fg_dir = root_dir + 'foregrounds/'
    bg_dir = root_dir + 'backgrounds/'
    invalid_fg_dir = root_dir + 'foregrounds_does_not_exist/'
    invalid_bg_dir = root_dir + 'backgrounds_does_not_exist/'

    test_config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'area': [0.3, 0.6],
        'rotation': [0],
        'bg_resolution': [(1000, 1000)]
    }
    with self.assertRaises(ValueError) as _:
      self.ds = dataset_generator.Dataset(
          test_config,
          new_dataset_dir=new_dataset_dir,
          foregrounds_dir=invalid_fg_dir,
          backgrounds_dir=bg_dir)

    with self.assertRaises(ValueError) as _:
      self.ds = dataset_generator.Dataset(
          test_config,
          new_dataset_dir=new_dataset_dir,
          foregrounds_dir=fg_dir,
          backgrounds_dir=invalid_bg_dir)

  def test_load_foregrounds(self):
    self.assertEqual(len(self.ds.fg_classes), self.num_test_fg_classes)
    self.assertEqual(sum(self.ds.num_fgs_per_class.values()), self.num_test_fgs)
    self.assertEqual(sum(self.ds.num_fgs_per_class_list), self.num_test_fgs)
    self.assertEqual(
        type(self.ds.fgs_dict['bagel'][0]), PIL.PngImagePlugin.PngImageFile)
    self.assertEqual(len(self.ds.fgs), self.num_test_fgs)

  def test_load_backgrounds(self):
    self.assertEqual(len(self.ds.bgs), self.num_test_bgs)
    self.assertEqual(self.ds.num_bgs, self.num_test_bgs)

  def test_generate_fg_bg_instance_tuples(self):
    fg_bg_tuples = self.ds._generate_fg_bg_instance_tuples()
    self.assertEqual(len(fg_bg_tuples),
                   self.num_test_fgs * self.ds.num_bgs_per_fg_instance)
    self.assertEqual(fg_bg_tuples[0][0], (0, 0))
    self.assertEqual(len(fg_bg_tuples[0]), self.ds.num_bgs_per_fg_instance)

  def test_preprocess_background_multiple_bg_sizes(self):
    """Checks that bg is not resized when there are multiple bg sizes."""
    self.ds.multiple_background_resolutions = True
    self.ds.bg_sizes = [(100, 100), (200, 200)]
    out = self.ds._preprocess_background(self.test_bg)
    self.assertEqual(self.test_bg.height, out.height)
    self.assertEqual(self.test_bg.width, out.width)

  def test_preprocess_background_one_bg_size(self):
    out = self.ds._preprocess_background(self.test_bg)
    self.assertEqual(out.height, 1000)
    self.assertEqual(out.width, 1000)

  def test_write_background_csv(self):
    csv_filepath = os.path.join(self.ds.new_dataset_dir, 'backgrounds.csv')
    with open(csv_filepath, 'r') as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        if i == 1:
          row_keep = row
    self.assertEqual(reader.line_num, self.num_test_bgs+1)
    self.assertEqual(int(row_keep[0]), 0)

  def test_generate_and_write_image_generate(self):
    image_metadata = {'id': 0,
                      'x_coord': 0.5,
                      'y_coord': 0.5,
                      'area': 0.3,
                      'rotation': 90,
                      'fg_class': 0,
                      'fg_instance': 0,
                      'bg_instance': 0,
                      'bg_resolution': (1000, 1000)}
    out = self.ds._generate_and_write_image(image_metadata)
    image_metadata.update({'pct_inside_image': 1})
    self.assertEqual(out, image_metadata)
    filepath = os.path.join(self.ds.new_dataset_dir, self.ds.fg_classes[0],
                            '0.jpg')
    self.assertTrue(os.path.exists(filepath))

  def test_generate_and_write_image_dont_generate(self):
    image_metadata = {'id': 2,
                      'x_coord': 1,
                      'y_coord': 0,
                      'area': 0.3,
                      'rotation': 90,
                      'fg_class': 0,
                      'fg_instance': 0,
                      'bg_instance': 0,
                      'bg_resolution': (1000, 1000)}
    out = self.ds._generate_and_write_image(image_metadata)
    self.assertIsNone(out)
    filepath = os.path.join(self.ds.new_dataset_dir, self.ds.fg_classes[1],
                            '2.jpg')
    self.assertFalse(os.path.exists(filepath))

  def test_write_batch_metadata(self):
    image_metadata = {'id': 2,
                      'x_coord': 1,
                      'y_coord': 0,
                      'area': 0.3,
                      'rotation': 90,
                      'fg_class': 0,
                      'fg_instance': 0,
                      'bg_instance': 0,
                      'bg_resolution': (1000, 1000),
                      'pct_inside_image': 0.95}
    batch_metadata = [image_metadata, image_metadata]
    self.ds._write_batch_metadata(batch_metadata)
    with open(self.ds.metadata_filepath, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        self.assertEqual(len(row), len(self.ds.metadata_header))
    self.assertEqual(reader.line_num, 2)

  def test_write_foreground_classes_csv(self):
    self.ds._write_foreground_classes_csv()
    csv_filepath = os.path.join(self.ds.new_dataset_dir,
                                'foreground_classes_metadata_indices.csv')
    with open(csv_filepath, 'r') as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        if i == 1:
          row_keep = row
    self.assertEqual(reader.line_num, self.num_test_fg_classes+1)
    self.assertEqual(int(row_keep[0]), 0)
    self.assertEqual(row_keep[1], 'bagel')

if __name__ == '__main__':
  unittest.main()

