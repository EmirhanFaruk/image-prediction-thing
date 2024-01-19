
# Modification date: Sun Apr 17 18:46:46 2022

# Production date: Sun Sep  3 15:43:43 2023

print("0")
import tflite_model_maker
print(tflite_model_maker.__version__)
#-----------------------------------------------------------------------------------------------------------------------
print("1")
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
#-----------------------------------------------------------------------------------------------------------------------
print("2")
#-----------------------------------------------------------------------------------------------------------------------
use_custom_dataset = True #@param ["False", "True"] {type:"raw"}

dataset_is_split = False #@param ["False", "True"] {type:"raw"}
#-----------------------------------------------------------------------------------------------------------------------
print("3")
#-----------------------------------------------------------------------------------------------------------------------
tf.__version__
#-----------------------------------------------------------------------------------------------------------------------
print("4")
#-----------------------------------------------------------------------------------------------------------------------
"""
# The ZIP file you uploaded:
!unzip image.zip
!unzip annotation.zip
"""

# Your labels map as a dictionary (zero is reserved):
label_map = {1: 'cloud', 2: 'water', 3:'ground'} 
"""
if dataset_is_split:
# If your dataset is already split, specify each path:
"""
train_images_dir = r'dataset\train\images'
train_annotations_dir = r'dataset\train\annotations'
val_images_dir = r'dataset\validation\images'
val_annotations_dir = r'dataset\validation\annotations'
test_images_dir = r'dataset\test\images'
test_annotations_dir = r'dataset\test\annotations'

#else:
# If it's NOT split yet, specify the path to all images and annotations
images_in = 'dataset\\images'
annotations_in = 'dataset\\annotations'
#-----------------------------------------------------------------------------------------------------------------------
print("5")
#-----------------------------------------------------------------------------------------------------------------------
import os
import random
import shutil

def split_dataset(images_path, annotations_path, val_split, test_split, out_path):
  """Splits a directory of sorted images/annotations into training, validation, and test sets.

  Args:
    images_path: Path to the directory with your images (JPGs).
    annotations_path: Path to a directory with your VOC XML annotation files,
      with filenames corresponding to image filenames. This may be the same path
      used for images_path.
    val_split: Fraction of data to reserve for validation (float between 0 and 1).
    test_split: Fraction of data to reserve for test (float between 0 and 1).
  Returns:
    The paths for the split images/annotations (train_dir, val_dir, test_dir)
  """
  _, dirs, _ = next(os.walk(images_path))

  train_dir = os.path.join(out_path, 'train')
  val_dir = os.path.join(out_path, 'validation')
  test_dir = os.path.join(out_path, 'test')

  IMAGES_TRAIN_DIR = os.path.join(train_dir, 'images')
  IMAGES_VAL_DIR = os.path.join(val_dir, 'images')
  IMAGES_TEST_DIR = os.path.join(test_dir, 'images')
  os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
  os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
  os.makedirs(IMAGES_TEST_DIR, exist_ok=True)

  ANNOT_TRAIN_DIR = os.path.join(train_dir, 'annotations')
  ANNOT_VAL_DIR = os.path.join(val_dir, 'annotations')
  ANNOT_TEST_DIR = os.path.join(test_dir, 'annotations')
  os.makedirs(ANNOT_TRAIN_DIR, exist_ok=True)
  os.makedirs(ANNOT_VAL_DIR, exist_ok=True)
  os.makedirs(ANNOT_TEST_DIR, exist_ok=True)

  # Get all filenames for this dir, filtered by filetype
  filenames = os.listdir(os.path.join(images_path))
  filenames = [os.path.join(images_path, f) for f in filenames if (f.endswith('.jpg'))]
  # Shuffle the files, deterministically
  filenames.sort()
  random.seed(42)
  random.shuffle(filenames)
  # Get exact number of images for validation and test; the rest is for training
  val_count = int(len(filenames) * val_split)
  test_count = int(len(filenames) * test_split)
  for i, file in enumerate(filenames):
    source_dir, filename = os.path.split(file)
    annot_file = os.path.join(annotations_path, filename.replace("jpg", "xml"))
    if i < val_count:
      shutil.copy(file, IMAGES_VAL_DIR)
      shutil.copy(annot_file, ANNOT_VAL_DIR)
    elif i < val_count + test_count:
      shutil.copy(file, IMAGES_TEST_DIR)
      shutil.copy(annot_file, ANNOT_TEST_DIR)
    else:
      shutil.copy(file, IMAGES_TRAIN_DIR)
      shutil.copy(annot_file, ANNOT_TRAIN_DIR)
  return (train_dir, val_dir, test_dir)
#-----------------------------------------------------------------------------------------------------------------------
print("6")
#-----------------------------------------------------------------------------------------------------------------------
train_dir, val_dir, test_dir = split_dataset(images_in, annotations_in,
                                                 val_split=0.2, test_split=0.2,
                                                 out_path='split-dataset')
train_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(train_dir, 'images'),
    os.path.join(train_dir, 'annotations'), label_map=label_map)
validation_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(val_dir, 'images'),
    os.path.join(val_dir, 'annotations'), label_map=label_map)
test_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(test_dir, 'images'),
    os.path.join(test_dir, 'annotations'), label_map=label_map)

print(f'train count: {len(train_data)}')
print(f'validation count: {len(validation_data)}')
print(f'test count: {len(test_data)}')
#-----------------------------------------------------------------------------------------------------------------------
print("7")
#-----------------------------------------------------------------------------------------------------------------------
spec = object_detector.EfficientDetLite0Spec()
#-----------------------------------------------------------------------------------------------------------------------
print("8")
#-----------------------------------------------------------------------------------------------------------------------
model = object_detector.create(train_data=train_data, 
                               model_spec=spec, 
                               validation_data=validation_data, 
                               epochs=50, 
                               batch_size=16, 
                               train_whole_model=True)
#-----------------------------------------------------------------------------------------------------------------------
print("9")
#-----------------------------------------------------------------------------------------------------------------------
model.evaluate(test_data)
#-----------------------------------------------------------------------------------------------------------------------
print("10")
#-----------------------------------------------------------------------------------------------------------------------
TFLITE_FILENAME = 'efficientdet-lite-bluelens.tflite'
LABELS_FILENAME = 'bluelens-labels.txt'
#-----------------------------------------------------------------------------------------------------------------------
print("11")
#-----------------------------------------------------------------------------------------------------------------------
model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
#-----------------------------------------------------------------------------------------------------------------------
print("12")
#-----------------------------------------------------------------------------------------------------------------------
model.evaluate_tflite(TFLITE_FILENAME, test_data)
#-----------------------------------------------------------------------------------------------------------------------
print("13")
#-----------------------------------------------------------------------------------------------------------------------
images_path = test_images_dir if dataset_is_split else os.path.join(test_dir, "images")
filenames = os.listdir(os.path.join(images_path))
random_index = random.randint(0,len(filenames)-1)
INPUT_IMAGE = os.path.join(images_path, filenames[random_index])
#-----------------------------------------------------------------------------------------------------------------------
print("14")
#-----------------------------------------------------------------------------------------------------------------------
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tflite_runtime.interpreter as tflite 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

def draw_objects(draw, objs, scale_factor, labels):
  """Draws the bounding box and label for each object."""
  COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)
  for obj in objs:
    bbox = obj.bbox
    color = tuple(int(c) for c in COLORS[obj.id])
    draw.rectangle([(bbox.xmin * scale_factor, bbox.ymin * scale_factor),
                    (bbox.xmax * scale_factor, bbox.ymax * scale_factor)],
                   outline=color, width=3)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
    draw.text((bbox.xmin * scale_factor + 4, bbox.ymin * scale_factor + 4),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color, font=font)

# Load the TF Lite model
labels = read_label_file(LABELS_FILENAME)
interpreter = tflite.Interpreter(TFLITE_FILENAME)
interpreter.allocate_tensors()

# Resize the image for input
image = Image.open(INPUT_IMAGE)
_, scale = common.set_resized_input(
    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

# Run inference
interpreter.invoke()
objs = detect.get_objects(interpreter, score_threshold=0.6, image_scale=scale)

# Resize again to a reasonable size for display
display_width = 500
scale_factor = display_width / image.width
height_ratio = image.height / image.width
image = image.resize((display_width, int(display_width * height_ratio)))
draw_objects(ImageDraw.Draw(image), objs, scale_factor, labels)
image
#-----------------------------------------------------------------------------------------------------------------------
print("15")
#-----------------------------------------------------------------------------------------------------------------------
objs