import numpy as np
import time
from scipy.misc import imresize
from scipy.ndimage import imread

from keras.models import Sequential
from keras.layers import Dense, Flatten, ELU, Convolution2D, Lambda, Dropout
from keras.callbacks import ModelCheckpoint

from skimage import transform
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb

import csv
import os
import json
import matplotlib.pyplot as plt

import tensorflow as tf
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 10, "Number of epochs")
flags.DEFINE_integer('samples_per_epoch', 128, "Batch size")
flags.DEFINE_string('model_name', 'model', "Name for saved model")
flags.DEFINE_string('training_dir', 'training', "Directory containing training data to be used")
flags.DEFINE_string('resume_model', '', "Saved weights to resume training")

EPOCHS = FLAGS.epochs
SAMPLES_PER_EPOCH = FLAGS.samples_per_epoch
MODEL_NAME = FLAGS.model_name
TRAINING_DIR = FLAGS.training_dir
RESUME_MODEL = FLAGS.resume_model

print("Running with arguments:")
print("Number of epochs:", EPOCHS)
print("Samples per epoch:", SAMPLES_PER_EPOCH)
print("Model name:", MODEL_NAME)
print("Training set:", TRAINING_DIR)
print("______________________________________________________________________")
print("")


def parse_driving_log(directory):
    """
    Reads in recorded data from simulator contained within specified directory
    """

    CENTER_IMAGE_COLUMN_IDX = 0
    LEFT_IMAGE_COLUMN_IDX = 1
    RIGHT_IMAGE_COLUMN_IDX = 2
    STEERING_COLUMN_IDX = 3

    STEERING_OFFSET = 0.15

    image_paths = []
    steering_angles = []

    val_image_paths = []
    val_steering_angles = []

    with open(directory + '/driving_log.csv') as csvfile:
        is_header = True
        reader = csv.reader(csvfile)

        for row in reader:
            if is_header:
                is_header = False
            else:
                steering_angle = float(row[STEERING_COLUMN_IDX])
                center_img = directory + '/' + row[CENTER_IMAGE_COLUMN_IDX].strip()

                # reserve 10% of center images for validation
                if random.randint(1, 10) == 1:
                    val_image_paths.append(center_img)
                    val_steering_angles.append(steering_angle)
                else:
                    image_paths.append(center_img)
                    steering_angles.append(steering_angle)

                    left_img = directory + '/' + row[LEFT_IMAGE_COLUMN_IDX].strip()
                    image_paths.append(left_img)
                    steering_angles.append(steering_angle + STEERING_OFFSET)

                    right_img = directory + '/' + row[RIGHT_IMAGE_COLUMN_IDX].strip()
                    image_paths.append(right_img)
                    steering_angles.append(steering_angle - STEERING_OFFSET)

    return image_paths, steering_angles, val_image_paths, val_steering_angles


def preprocess_image(img):
    """
    Converts image to hsv color space, crops out sky and car's dashboard, and resizes image.
    """
    # convert to hsv
    img = rgb2hsv(img)

    # crop top of image which shows only the sky and crop bottom portion where part of the car's dashboard is
    height = img.shape[0]
    img = img[int(height / 4): height - int(height / 8), :]

    img = imresize(img, (66, 200))

    return img


def generate_samples(image_paths, steering_angles, batch_size=256):
    """
    Generates samples for training by loading image and
        - performing random shifts left and right
        - randomly adjusting brightness
        - converting image to hsv color space and cropping
        - flipping image
    """
    batch_size = min(len(image_paths), batch_size)
    assert (len(image_paths) == len(steering_angles))

    X = np.zeros((batch_size, 66, 200, 3), dtype=np.uint8)
    y = np.zeros(batch_size, dtype=np.float64)

    while True:

        batch_idx = 0
        while batch_idx < batch_size:
            idx = random.randrange(len(image_paths))

            img = imread(image_paths[idx])
            steering_angle = steering_angles[idx]

            # apply x,y translation to 40% of images
            if random.randint(0, 10) < 3:
                img, steering_angle = shift(img, steering_angle, shift_scale=0.2)

            # random brightness adjustment between 50% and 125% of original brightness
            img = random_brightness_adjustment(img, min_brightness_scale=0.25, max_brightness_scale=1.25)

            img = preprocess_image(img)

            X[batch_idx] = img
            y[batch_idx] = steering_angle
            batch_idx += 1

            if batch_idx < batch_size - 1:
                X[batch_idx] = np.fliplr(img)
                y[batch_idx] = -steering_angle
                batch_idx += 1

        yield X, y


def shift(img, steering_angle, shift_scale=.1):
    """
    Shift image to the left and adjust steering angle based off of shift_scale value
    """
    # shift as scale of images width and height
    x_shift = int(random.uniform(-shift_scale, shift_scale) * img.shape[1])

    # artificially adjust steering to account for shift
    steering_angle += (x_shift * -0.0033)

    t = transform.SimilarityTransform(scale=1.0, translation=(x_shift, 0))
    warped_img = transform.warp(img, t)
    return warped_img, steering_angle


def random_brightness_adjustment(img, min_brightness_scale=0.5, max_brightness_scale=1.1):
    """
    Randomly adjusts brightness of rgb image based off of the min_brightness_scale and max_brightness_scale values
    """
    adjusted_img = rgb2hsv(img)

    random_adjustment = random.uniform(min_brightness_scale, max_brightness_scale)

    adjusted_img[:, :, 2] = adjusted_img[:, :, 2] * random_adjustment
    adjusted_img = np.clip(adjusted_img, 0, 1)

    return hsv2rgb(adjusted_img)


def build_model(input_shape):
    """
    Convolutional Neural network used for training.  Based of of NVIDIA's architecture from the paper found here at
     http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()

    # normalize between -1 and 1
    model.add(Lambda(lambda x: x / 127.5, input_shape=input_shape))

    # convolution layers using Gaussian initialization
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))

    model.add(ELU())
    model.add(Flatten())

    # fully connected
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    if RESUME_MODEL != '':
        model.load_weights(RESUME_MODEL)

    return model


def save_model(model, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    model.save_weights(dir + '/' + FLAGS.model_name + '.h5', True)
    with open(dir + '/' + FLAGS.model_name + '.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


def main():
    # load images
    image_paths, steering_angles, val_image_paths, val_steering_angles = run('parsing driving log', parse_driving_log)(
        TRAINING_DIR)

    print("")
    print("running with", len(image_paths), "images dedicated to training,", len(val_image_paths),
          "images dedicated to validation")
    print("")

    x, y = next(generate_samples(image_paths[0:10], steering_angles[0:10]))

    # build and compile model
    model = run('building model', build_model)(x[0].shape)

    # compile model using adam optimizer
    run('compiling model', model.compile)("adam", "mse")

    weight_save_callback = ModelCheckpoint('saved_weights/weights.{epoch:02d}_{val_loss:.3f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=False, mode='auto')

    # train model
    run('training model', model.fit_generator)(generate_samples(image_paths, steering_angles),
                                               samples_per_epoch=SAMPLES_PER_EPOCH,
                                               nb_epoch=EPOCHS,
                                               validation_data=generate_samples(val_image_paths,
                                                                                val_steering_angles),
                                               nb_val_samples=1000,
                                               callbacks=[weight_save_callback])

    # save trained model
    run('saving model', save_model)(model, '.')


def run(task_name, fn):
    def fn_wrap(*args, **kwargs):
        print("%s..." % task_name)
        start_time = time.time()
        retval = fn(*args, **kwargs)
        print('Finished %s in %d seconds.' % (task_name, time.time() - start_time))
        print('')
        return retval

    return fn_wrap


if __name__ == "__main__":
    main()
