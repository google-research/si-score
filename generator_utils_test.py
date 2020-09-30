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

# Lint as: python3
"""Tests for methods in `dataset_generator.py`.

Tests methods that are not in the Dataset class.
`generator_test.py` tests the methods that are in the Dataset class.

Run tests using the command `python -m unittest generator_utils_test.py`.
"""

import unittest

import dataset_generator
import numpy as np
import time


class DatasetTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # TODO: update dir
    root_dir = './'
    current_time = int(time.time())
    new_dataset_dir = f'/tmp/test_dir/{current_time}/'
    fg_dir = root_dir + 'foreground_samples/'
    bg_dir = root_dir + 'background_samples/'  # max 1000px per side
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
        backgrounds_dir=bg_dir)
    self.test_fg = self.ds.fgs_dict['bagel'][0]
    test_bg = dataset_generator.load_image('background_samples/beach2.jpg')
    self.test_bg = self.ds._preprocess_background(test_bg)

  def test_generate_instance_tuples(self):
    test_list = [1, 1, 2, 3]
    out = dataset_generator.generate_instance_tuples(test_list)
    self.assertEqual(len(out), 1+1+2+3)
    self.assertEqual(out[0], (0, 0))
    self.assertEqual(out[3], (2, 1))

  def test_crop_image_to_square(self):
    out = dataset_generator.crop_image_to_square(self.test_bg)
    self.assertEqual(out.width, out.height)

  def test_resize_fg(self):
    fg_target_size = 0.5  # target is fg area = 50% bg area
    resized_fg = dataset_generator.resize_fg(self.test_fg, self.test_bg,
                                             fg_target_size)
    ratio = resized_fg.height * resized_fg.width / (
        self.test_bg.width * self.test_bg.height)
    self.assertAlmostEqual(ratio, fg_target_size, delta=0.05)

  def test_resize_fg_too_large(self):
    fg_target_size = 3  # target is fg area = 300% bg area
    resized_fg = dataset_generator.resize_fg(self.test_fg, self.test_bg,
                                             fg_target_size)
    ratio = resized_fg.height * resized_fg.width / (
        self.test_bg.width * self.test_bg.height)
    self.assertLessEqual(ratio, 1)
    max_fg_length_ratio = max(resized_fg.height / self.ds.bg_sizes[0][0],
                              resized_fg.width / self.ds.bg_sizes[0][1])
    self.assertGreaterEqual(max_fg_length_ratio, 0.99)

  def test_paste_fg_on_bg(self):
    pasted_img = dataset_generator.paste_fg_on_bg(
        self.test_fg, self.test_bg, x_coord=0, y_coord=0)
    # Mediocre test: number obtained from a colab
    # where the results looked correct.
    self.assertAlmostEqual(np.mean(pasted_img), 148.089, delta=0.1)

  def test_resize_bg(self):
    out = dataset_generator.resize_bg(
        self.test_bg, tgt_width=500, tgt_height=500)
    self.assertEqual(out.height, 500)
    self.assertEqual(out.width, 500)

  def test_resize_bg_not_square(self):
    out = dataset_generator.resize_bg(
        self.test_bg, tgt_width=100, tgt_height=200)
    self.assertEqual(out.height, 200)
    self.assertEqual(out.width, 100)

  def test_calc_top_left_coordinates(self):
    x_start, y_start = dataset_generator.calc_top_left_coordinates(
        self.test_fg, self.test_bg, 0.5, 0.5)
    self.assertEqual(x_start, 141)
    self.assertEqual(y_start, 158)

  def test_calc_pct_inside_image_100pct(self):
    out1 = dataset_generator.calc_pct_inside_image(self.test_fg, self.test_bg,
                                                   0, 0)
    self.assertEqual(out1, 1)

  def test_calc_pct_inside_image_50pct(self):
    out2 = dataset_generator.calc_pct_inside_image(self.test_bg, self.test_bg,
                                                   0, 500)
    self.assertEqual(out2, 0.5)

  def test_rotate_image(self):
    out = dataset_generator.rotate_image(self.test_fg, 11)
    self.assertAlmostEqual(np.mean(out), 77.633, delta=0.1)
    cos_angle = np.cos(11 * np.pi / 180)
    sin_angle = np.sin(11 * np.pi / 180)
    new_width = int(self.test_fg.width * cos_angle +
                    self.test_fg.height * sin_angle)
    new_height = int(self.test_fg.height * cos_angle +
                     self.test_fg.width * sin_angle)
    self.assertAlmostEqual(out.height, new_height, delta=2)
    self.assertAlmostEqual(out.width, new_width, delta=2)

  def test_validate_config(self):
    config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'area': [0.3, 0.6],
        'rotation': [0],
        'bg_resolution': (1000, 1000)
    }
    with self.assertRaises(TypeError) as _:
      dataset_generator.validate_config(config)

  def test_validate_config_missing_coord_field(self):
    config = {
        'area': [0.3, 0.6],
        'rotation': [0],
        'bg_resolution': [(1000, 1000)]
    }
    with self.assertRaises(ValueError) as _:
      dataset_generator.validate_config(config)

  def test_validate_config_missing_area_field(self):
    config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'rotation': [0],
        'bg_resolution': [(1000, 1000)]
    }
    with self.assertRaises(ValueError) as _:
      dataset_generator.validate_config(config)

  def test_validate_config_missing_rotation_field(self):
    config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'area': [0.3, 0.6],
        'bg_resolution': [(1000, 1000)]
    }
    with self.assertRaises(ValueError) as _:
      dataset_generator.validate_config(config)

  def test_validate_config_missing_bg_res_field(self):
    config = {
        'coord': [(0.0, 0.0), (0.5, 0.5)],
        'area': [0.3, 0.6],
        'rotation': [0],
    }
    with self.assertRaises(ValueError) as _:
      dataset_generator.validate_config(config)


if __name__ == '__main__':
  unittest.main()

