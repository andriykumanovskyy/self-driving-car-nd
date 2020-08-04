import csv
import cv2
import numpy as np


lines = []
images = []
measurements = []
with open('C:\Projects\CarND-Behavioral-Cloning-P3\data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
        
    for line in reader:
        line.append(line)
        ## controls the left/right/centre (3)images, change to 1 for just centre. Also, to avoid correction factor, centre to mid 
        # driving data collection do, but jot down how to do it for future
        for i in range(1):
            source_path = line[i]
            #print(source_path)
            #print('')
            image = cv2.imread(source_path) 
            images.append(image)    
            measurement = float(line[3])
            measurements.append(measurement)
            
    for line in reader:
        line.append(line)
    
        for i in range(2):
            source_path = line[i]

            l_image = cv2.imread(source_path)
            images.append(l_image)    
            l_measurement = float(line[9])
            measurements.append(l_measurement)
            
    for line in reader:
        line.append(line)
    
        for i in range(3): 
            source_path = line[i]
            # try 1 on all of them (where there are 2 and 3 now)
            r_image = cv2.imread(source_path)
            images.append(r_image)    
            r_measurement = float(line[10])
            measurements.append(r_measurement)
            
#need to make this augmentation work with the extra data on left and right cameras.. meaning not just image, use l_image, r_image ect. 
# and the methods on the paper written by bed... like resizing images

#augmented_images, augmented_measurements = [], []
#for image,measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(images)
y_train = np.array(measurements)
print(X_train)
print(y_train)

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Conv2D
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, SpatialDropout2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers.core import Lambda


from keras.optimizers import SGD, Adam, RMSprop
# from keras import backend as K
# K.set_image_dim_ordering('tf')


batch = 128
epochs = 7
activation_type = 'relu'
dropout_p = .3

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
### model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
### model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
### model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Convolution2D(64,3,3, activation="relu"))
### model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Convolution2D(64,3,3, activation="relu"))
### model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Flatten())
model.add(Dense(1162, activation=activation_type))
model.add(Dense(100, activation=activation_type))
model.add(Dense(50, activation=activation_type))
model.add(Dense(10, activation=activation_type))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')