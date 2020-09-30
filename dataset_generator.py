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
"""Class to generate synthetic dataset.

This is used in `generator_script.py`.
Editing that script is probably the easiest way to generate 
a synthetic dataset.

Example of how to use this class:
```
import dataset_generator as synthetic

import tensorflow.io.gfile as gfile
from os import path

config = {
    'coord': [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
    'area': [0.2],
    'rotation': [0],
    'bg_resolution': [(500, 500)],
}
dataset_name = 'test'

dataset_dir = './test/'
new_dataset_dir = path.join(dataset_dir, dataset_name, '')

if not gfile.exists(new_dataset_dir):
  gfile.makedirs(new_dataset_dir)

dataset = synthetic.Dataset(
    config=config,
    new_dataset_dir=new_dataset_dir,
    num_bgs_per_fg_instance=2)

dataset.generate_dataset()
```
"""
import csv
import functools
import io
import itertools
from multiprocessing import pool
import operator
from os import path
import label_str_to_imagenet_classes as label_dict
import numpy as np
import PIL
from PIL import Image

import tensorflow.io.gfile as gfile

ROOT_DIR = '.'

DEFAULT_FG_DIR = path.join(ROOT_DIR, 'foregrounds', '')
DEFAULT_BG_DIR = path.join(ROOT_DIR, 'backgrounds', '')


def load_image(fname):
  with open(fname, mode='rb') as f:
    s = f.read()

  image = Image.open(io.BytesIO(s))
  image.load()
  return image


def validate_config(config):
  """Checks config file has all required fields.

  Raises an error if this is not the case.

  Args:
    config: dict.

  Returns:
    config: dict. (unchanged)
  """
  if 'area' not in config.keys():
    raise ValueError('Must specify area.')
  if 'coord' not in config.keys():
    raise ValueError('Must specify coordinates `coord`.')
  if 'rotation' not in config.keys():
    raise ValueError('Must specify rotation angle `rotation`.')
  if 'bg_resolution' not in config.keys():
    raise ValueError('Must specify bg_resolution.')
  bg_res = config['bg_resolution']
  if not isinstance(bg_res, list) or not isinstance(bg_res[0], tuple) or len(
      bg_res[0]) != 2:
    raise TypeError('bg_resolution should be a list of tuples (width, height).')
  return config


def resize_fg(fg, bg, fg_target_size):
  """Resizes foregrounds to `fg_target_size`% of the background area."""
  # Resize foreground to have area = fg_size**2 * background_area.
  fg_copy = fg.copy()
  fg_area = fg.height * fg.width
  bg_area = bg.width * bg.height
  fg_area_ratio = fg_area / bg_area
  resize_factor = np.sqrt(fg_target_size / fg_area_ratio)
  fg_copy = fg_copy.resize(
      (int(fg.width * resize_factor), int(fg.height * resize_factor)),
      PIL.Image.BILINEAR)
  return fg_copy


def paste_fg_on_bg(fg, bg, x_coord, y_coord):
  """Pastes foreground on background at offset (x_coord, y_coord).

  x_coord, y_coord are floats in range [0, 1].

  Args:
    fg: foreground image of type PIL image. Examples of PIL image types include
      PIL.PngImagePlugin.PngImageFile and PIL.JpegImagePlugin.JpegImageFile.
    bg: background image of type PIL image.
    x_coord: float in range [0, 1]. x-coord offset from top left, for pasting
    foreground.
    y_coord: float in range [0, 1]. y-coord offset from top left, for pasting
    foreground.

  Returns:
    Background: PIL image (e.g. type PIL.JpegImagePlugin.JpegImageFile).
  """
  bg_copy = bg.copy()
  bg_copy.paste(fg, box=(x_coord, y_coord), mask=fg)
  return bg_copy


def resize_bg(bg, tgt_width, tgt_height):
  """Resizes bg to width = tgt_width, height = tgt_height."""
  return bg.resize((tgt_width, tgt_height), PIL.Image.BILINEAR)


def crop_image_to_square(img):
  """Crops image to the largest square that fits inside img.

  Crops from the top left corner.

  Args:
    img: image of type PIL image, e.g. PIL.JpegImagePlugin.JpegImageFile.

  Returns:
    Square image of same type as input image.
  """
  side_length = min(img.height, img.width)
  return img.crop((0, 0, side_length, side_length))


def calc_top_left_coordinates(fg, bg, x_coord, y_coord):
  """Returns coordinates of top left corner of object.

  Input coordinates are coordinates of centre of object scaled in the range
  [0, 1].

  Args:
    fg: PIL image. Foreground image.
    bg: PIL image. Background image.
    x_coord: central x-coordinate of foreground object scaled between 0 and 1.
      0 = leftmost coordinate of image, 1 = rightmost coordinate of image.
    y_coord: central y-coordinate of foreground object scaled between 0 and 1.
      0 = topmost coordinate of image, 1 = bottommost coordinate of image.
  """
  x_coord = int(x_coord * bg.width)
  y_coord = int(y_coord * bg.height)
  # x_coord, y_coord should be at the centre of the object.
  x_coord_start = int(x_coord - fg.width*0.5)
  y_coord_start = int(y_coord - fg.height*0.5)

  return x_coord_start, y_coord_start


def calc_pct_inside_image(fg, bg, x_coord_start, y_coord_start):
  """Calculate the percentage of the object that is inside the image.

  This calculation is based on the bounding box of the object
  as opposed to object pixels.

  Args:
    fg: PIL image. Foreground image.
    bg: PIL image. Background image.
    x_coord_start: leftmost x-coordinate of foreground object.
    y_coord_start: topmost y-coordinate of foreground object.

  Returns:
    Float between 0.0 and 1.0 inclusive, indicating the percentage of the
    object that is in the image.
  """
  x_coord_end = x_coord_start + fg.width
  y_coord_end = y_coord_start + fg.height

  x_obj_start = max(x_coord_start, 0)
  x_obj_end = min(x_coord_end, bg.width)
  y_obj_start = max(y_coord_start, 0)
  y_obj_end = min(y_coord_end, bg.height)

  object_area = fg.width * fg.height
  area_inside_image = (x_obj_end - x_obj_start) * (y_obj_end - y_obj_start)
  pct_inside_image = area_inside_image / object_area
  return pct_inside_image


def generate_instance_tuples(num_per_class_list):
  """Generate list of tuples [(class_index, instance_index)...]."""
  num_classes = len(num_per_class_list)
  class_and_instance_indices = []
  for i in range(num_classes):
    num_instances = num_per_class_list[i]
    class_and_instance_indices.extend([(i, j) for j in range(num_instances)])

  return class_and_instance_indices


def rotate_image(img, rotation_angle):
  """Rotate image by rotation_angle counterclockwise."""
  return img.rotate(
      rotation_angle, resample=PIL.Image.BICUBIC, expand=True)


def write_backgrounds_csv(new_dataset_dir, backgrounds_dir):
  """Write CSV mapping background ints to bg filenames."""
  bg_filenames = gfile.glob(path.join(backgrounds_dir, '*'))
  bg_filenames = [fname.split('/')[-1] for fname in bg_filenames]
  csv_filepath = path.join(new_dataset_dir, 'backgrounds.csv')
  with open(csv_filepath, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['int', 'label'])

    for i, fname in enumerate(bg_filenames):
      writer.writerow([i, fname])


class Dataset:
  """Generates 2.5D synthetic data and saves it to `new_dataset_dir`.

  Use the `Dataset.generate_dataset()` method to generate the dataset and
  save it to `new_dataset_dir`.
  """

  def __init__(self,
               config,
               new_dataset_dir,
               foregrounds_dir=DEFAULT_FG_DIR,
               backgrounds_dir=DEFAULT_BG_DIR,
               num_bgs_per_fg_instance=2,
               min_pct_inside_image=0.95):
    """Initialise dataset.

    Args:
      config: ConfigDict.
        Sample config: {'coords':[0.5, 0.5],
                        'area':[0.5],
                        'rotation':[0],
                        'bg_resolution:(1000,1000)}
      new_dataset_dir: string, directory to save images in.
      foregrounds_dir: string, path to foregrounds directory.
        directory has form `foregrounds_dir/{class_name}/{image.jpg}`.
      backgrounds_dir: string, path to backgrounds directory.
        directory has form `backgrounds_dir/{bg_name}.jpg`.
      num_bgs_per_fg_instance: int, number of background images sampled to be
        combined with each foreground object instance. Max value = number of
        backgrounds provided.
      min_pct_inside_image: float in [0, 1], minimum percentage of object that
        needs to be inside the image. If a generated image
        does not meet this criteria, it is not included.
    """
    self.config = validate_config(config)
    self.new_dataset_dir = new_dataset_dir
    self.num_bgs_per_fg_instance = num_bgs_per_fg_instance
    self.min_pct_inside_image = min_pct_inside_image
    self.bg_sizes = self.config['bg_resolution']  # width, height
    self.multiple_background_resolutions = False
    if len(self.bg_sizes) > 1:
      self.multiple_background_resolutions = True
    self.metadata_filepath = path.join(new_dataset_dir, 'metadata.csv')
    self.metadata_header = [
        'image_id', 'x_coord', 'y_coord', 'area', 'rotation',
        'foreground_class', 'foreground_instance', 'background',
        'bg_resolution_width', 'bg_resolution_height', 'pct_inside_image'
    ]
    self._thread_pool = pool.ThreadPool(100)

    self._load_foregrounds(foregrounds_dir)
    self._load_backgrounds(backgrounds_dir)

    self._make_root_class_dirs()
    write_backgrounds_csv(self.new_dataset_dir, backgrounds_dir)

  def _generate_image_metadata(self):
    """Generates image metadata as cartesian product of attributes.

    Uses self.config and sets metadata as self.image_metadata.
    """
    fgs_bgs = self._generate_fg_bg_instance_tuples()

    config_lists = [self.config['coord'],
                    self.config['area'],
                    self.config['rotation'],
                    fgs_bgs,
                    self.config['bg_resolution']
                   ]

    image_metadata = itertools.product(*config_lists)

    self.num_images = functools.reduce(operator.mul, map(len, config_lists), 1)
    self.image_metadata = []
    for i, row in enumerate(image_metadata):
      temp_dict = {'id': i,
                   'x_coord': row[0][0],
                   'y_coord': row[0][1],
                   'area': row[1],
                   'rotation': row[2],
                   'fg_class': row[3][0][0],
                   'fg_instance': row[3][0][1],
                   'bg_instance': row[3][1],
                   'bg_resolution': row[4]}
      self.image_metadata.append(temp_dict)

  def _preprocess_background(self, bg):
    """Crops background to a square and resizes background if applicable.

    Only resizes background if exactly one background resolution is provided
    in the config.

    Args:
      bg: background of type PIL image, e.g. PIL.JpegImagePlugin.JpegImageFile.

    Returns:
      Resized background of type PIL.Image.Image.
    """
    bg = crop_image_to_square(bg)
    # If only one bg size is given [(width, height)]
    if not self.multiple_background_resolutions:
      if bg.width != self.bg_sizes[0][0] or bg.height != self.bg_sizes[0][1]:
        bg = bg.resize((self.bg_sizes[0][0], self.bg_sizes[0][1]),
                       PIL.Image.BILINEAR)
    return bg

  def _load_foregrounds(self, foregrounds_dir):
    """Loads foregrounds from a directory.

    Args:
      foregrounds_dir: path to directory containing foregrounds.
        Directory of the form `foregrounds_dir`/$OBJECT_CLASS/$FILE_NAME.

    Produces:
      self.fg_classes: a list of names of foreground object classes, e.g.
        ['ambulance', 'bagel', ...]
      self.num_fgs_per_class: a dict of the form {foreground_obj_class_name:
        num_fgs_in_that_class}
      self.fgs: a list of the form [fg0, fg1, ...] where the foregrounds are
        `PIL.PngImagePlugin.PngImageFile`s.
      self.fgs_dict: a dict of the form {fg_class_name: [img0, img1, ...]} where
        the images are `PIL.PngImagePlugin.PngImageFile`s.
    """
    if not gfile.exists(foregrounds_dir):
      raise ValueError(
          f'Foregrounds directory {foregrounds_dir} does not exist.')
    fg_fnames = gfile.glob(path.join(foregrounds_dir, '*/*'))
    fg_labels = [x.split('/')[-2] for x in fg_fnames]  # e.g. 'car', 'cow'
    self.fg_classes = sorted(list(set(fg_labels)))
    self.num_fgs_per_class = {
        fg_class: len(gfile.glob(path.join(foregrounds_dir, fg_class, '*')))
        for fg_class in self.fg_classes
    }
    self.num_fgs_per_class_list = [
        self.num_fgs_per_class[fg_class] for fg_class in self.fg_classes
    ]
    self.fgs = self._thread_pool.map(load_image, fg_fnames)
    self.fgs_dict = {fg_class: [] for fg_class in self.fg_classes}
    for i, label in enumerate(fg_labels):
      self.fgs_dict[label].append(self.fgs[i])

    print('Foregrounds loaded.')

  def _load_backgrounds(self, backgrounds_dir):
    """Loads backgrounds from a directory.

    Args:
      backgrounds_dir: path to directory containing foregrounds.
        Dir of the form `backrounds_dir`/$BACKGROUND_TYPE/$FILE_NAME.

    Produces:
      self.bgs: a list of the form [bg0, bg1, ...] where the backgrounds
        are `PIL.Image.Image`s.
      self.num_bgs: int, number of backgrounds.
    """
    if not gfile.exists(backgrounds_dir):
      raise ValueError(
          f'Backgrounds directory {backgrounds_dir} does not exist.')
    bg_fnames = gfile.glob(path.join(backgrounds_dir, '*'))
    self.bgs = self._thread_pool.map(load_image, bg_fnames)
    self.bgs = self._thread_pool.map(self._preprocess_background, self.bgs)
    self.num_bgs = len(self.bgs)

    print('Backgrounds loaded.')

  def _make_root_class_dirs(self):
    """Make dataset root dir and subdir for each class."""
    if not gfile.exists(self.new_dataset_dir):
      gfile.makedirs(self.new_dataset_dir)

    for class_name in self.fg_classes:
      class_dir = path.join(self.new_dataset_dir, class_name, '')
      if not gfile.exists(class_dir):
        gfile.mkdir(class_dir)

  def _generate_and_write_images_and_metadata(self, batch_size=128):
    """Generate and save images and write metadata for a dataset."""
    metadata_batch = []
    for i, single_image_metadata in enumerate(self.image_metadata):
      metadata_batch.append(single_image_metadata)

      if (i + 1) % batch_size == 0:
        self._generate_and_write_images_and_metadata_batch(metadata_batch)
        metadata_batch = []
    self._generate_and_write_images_and_metadata_batch(metadata_batch)

  def _generate_and_write_images_and_metadata_batch(self, metadata_batch):
    """Generate and save images and write metadata for a batch of images."""
    metadata_batch = self._thread_pool.map(self._generate_and_write_image,
                                           metadata_batch)
    self._write_batch_metadata(metadata_batch)

  def _generate_fg_bg_instance_tuples(self):
    """Generate tuples with fg and bg instances to describe imgs to generate.

    Each tuple is ((fg_class_int, fg_instance_int), bg_instance_int).

    Returns:
      List of such tuples.
    """
    fg_tuples = generate_instance_tuples(self.num_fgs_per_class_list)
    fg_bg_tuples = []
    for fg_tuple in fg_tuples:
      bgs = np.random.choice(
          self.num_bgs, self.num_bgs_per_fg_instance, replace=False)
      fg_bg_tuples.extend([(fg_tuple, bg) for bg in bgs])
    return fg_bg_tuples

  def generate_dataset(self, batch_size=128):
    """Generate and write synthetic dataset and associated metadata."""
    self._generate_image_metadata()
    self._write_metadata_header()
    self._write_foreground_classes_csv()
    self._write_foreground_classes_imagenet_ints_csv()
    self._generate_and_write_images_and_metadata(batch_size=batch_size)
    print('Dataset generated.')

  def _generate_and_write_image(self, image_metadata):
    """Generate image from metadata and write to dataset directory."""
    fg_class, fg_instance = image_metadata['fg_class'], image_metadata[
        'fg_instance']
    bg_num = image_metadata['bg_instance']
    x_coord, y_coord = image_metadata['x_coord'], image_metadata['y_coord']
    fg_tgt_area = image_metadata['area']
    rotation_angle = image_metadata['rotation']
    bg_resolution = image_metadata['bg_resolution']

    fg = self.fgs_dict[self.fg_classes[fg_class]][fg_instance]
    bg = self.bgs[bg_num]

    if self.multiple_background_resolutions:
      bg = resize_bg(bg, bg_resolution[0], bg_resolution[1])
    fg = resize_fg(fg, bg, fg_tgt_area)  # fg_target_size, uses background size
    fg = rotate_image(fg, rotation_angle)
    x_coord_start, y_coord_start = calc_top_left_coordinates(
        fg, bg, x_coord, y_coord)
    pct_inside_image = calc_pct_inside_image(fg, bg, x_coord_start,
                                             y_coord_start)
    if pct_inside_image < self.min_pct_inside_image:
      return None
    pct_inside_image = round(pct_inside_image, 4)
    image = paste_fg_on_bg(fg, bg, x_coord_start, y_coord_start)

    # write image to directory
    image_id = image_metadata['id']
    label = self.fg_classes[fg_class]
    file_path = '{}/{}/{}.jpg'.format(self.new_dataset_dir, label, image_id)
    with open(file_path, 'wb') as f:
      try:
        image.save(f)
      except TypeError:
        print('Failed to generate image num {}: fg: {} {}, bg: {}'.format(
            image_id, fg_class, fg_instance, bg_num))
        print('Problem is likely that one of the foreground or background '
              'images has an incompatible format. Loading and saving them as '
              'JPG images using PIL may solve this problem.')

    image_metadata.update(
        {'pct_inside_image': pct_inside_image})

    return image_metadata

  def _write_metadata_header(self):
    """Writes metadata header to self.metadata_filepath."""
    with open(self.metadata_filepath, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(self.metadata_header)

  def _write_batch_metadata(self, batch_metadata):
    """Appends batch of metadata to self.metadata_filepath.

    Args:
      batch_metadata: list of lists. Each list is of the form
        [(x_coord, y_coord), area, rotation_angle, ((foreground_class_int),
        foreground_instance_int), background_instance_int),
        (bg_width, bg_height)],
        or is `None` if pct_inside_image < min_pct_inside_image.
    """
    with open(self.metadata_filepath, 'a') as f:
      writer = csv.writer(f)

      for row in batch_metadata:
        # `row` is None if pct_inside_image is not large enough
        if row is not None:
          csv_row = [row['id']]  # image ID, int
          csv_row.extend([row['x_coord'], row['y_coord']])
          csv_row.extend([row['area'], row['rotation']])  # area, rotation
          csv_row.extend(
              [row['fg_class'], row['fg_instance'], row['bg_instance']])
          bg_resolution = row['bg_resolution']
          csv_row.extend([bg_resolution[0], bg_resolution[1]])
          csv_row.extend([row['pct_inside_image']])
          writer.writerow(csv_row)

  def _write_foreground_classes_csv(self):
    """Write CSV that lists foreground class names and integer indices.

    Indices are those used in metadata.
    """
    csv_filepath = path.join(self.new_dataset_dir,
                             'foreground_classes_metadata_indices.csv')
    with open(csv_filepath, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['int', 'label'])

      for i, label in enumerate(self.fg_classes):
        writer.writerow([i, label])

  def _write_foreground_classes_imagenet_ints_csv(self):
    """Write CSV that lists foreground class names and integer indices.

    Indices are those used in metadata.
    """
    csv_filepath = path.join(self.new_dataset_dir, 'foreground_classes.csv')
    with open(csv_filepath, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['int', 'label'])

      for label in self.fg_classes:
        index = label_dict.label_str_to_imagenet_classes[label]
        writer.writerow([index, label])
