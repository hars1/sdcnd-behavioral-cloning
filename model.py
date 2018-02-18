import cv2
import csv
import numpy as np
import os

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

#################################################################

def read_data_from_csv(dataPath, skipHeader=False):
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)

    return lines

def read_images_and_measurements(dataPath):
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerImages = []
    leftImages = []
    rightImages = []
    measurementsImages = []
    for directory in dataDirectories:
        lines = read_data_from_csv(directory, skipHeader=True)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerImages.extend(center)
        leftImages.extend(left)
        rightImages.extend(right)
        measurementsImages.extend(measurements)

    return (centerImages, leftImages, rightImages, measurementsImages)

def augment_measurements_and_combine_images(centerImages, leftImages, rightImages, measurement, correction):
    images = []
    images.extend(centerImages)
    images.extend(leftImages)
    images.extend(rightImages)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])

    return (images, measurements)

#################################################################

def cnn_model():
    # Difference to the Nvidia Model is the cropping and the dropout layers.
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    # normalize data.
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(24,(5,5), subsample=(2,2), activation='relu', border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Conv2D(36,(5,5), subsample=(2,2), activation='relu', border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Conv2D(48,(5,5), subsample=(2,2), activation='relu', border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Conv2D(64,(3,3), activation='relu', border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Conv2D(64,(3,3), activation='relu', border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.summary()
    return model

#################################################################

def generate_data(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping image, correcting measurement and adding that measuerement.
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

################################################################

# Reading images (center, right, left) and measurements.
centerImages, leftImages, rightImages, measurements = read_images_and_measurements('data')
images, measurements = augment_measurements_and_combine_images(centerImages, leftImages, rightImages, measurements, 0.2)
print('Total Images: {}'.format( len(images)))

# Splitting samples and creating generators.
samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

#################################################################

training_generator = generate_data(train_samples, batch_size=32)
validation_generator = generate_data(validation_samples, batch_size=32)

print('Training model...')

model = cnn_model()
model.compile(optimizer=Adam(lr=1e-4), loss='mse')


history_object = model.fit_generator(training_generator, steps_per_epoch= \
                 2*len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=2*len(validation_samples), nb_epoch=2, verbose=1)

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

#################################################################

print('Saving model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")
