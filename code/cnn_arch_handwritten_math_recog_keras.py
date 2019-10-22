# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

batch_size = 64
num_classes = 67
epochs = 30
input_shape = (28, 28, 1)
datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,
        validation_split=0.2) # randomly flip images
train_generator = datagen.flow_from_directory(
    directory=r"/kaggle/input/extracted_data_train_test/extracted_data_train_test/train_imgs2/",
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    directory=r"/kaggle/input/extracted_data_train_test/extracted_data_train_test/train_imgs2/", # same directory as training data
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset='validation')
x_train, y_train = next(train_generator)
print(x_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
model.summary()

#Architencture2
#model = Sequential()

#model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                 activation ='relu', input_shape = (28,28,1)))
#model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))


#model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#               activation ='relu'))
#model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.25))


#model.add(Flatten())
#model.add(Dense(256, activation = "relu"))
#model.add(Dropout(0.5))
#model.add(Dense(67, activation = "softmax"))
#optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                            patience=3, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0.00001)
#model.summary()

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    target_size=(28, 28),
    batch_size = 64,
    color_mode="grayscale",
    shuffle = False,
    directory=r"/kaggle/input/extracted_data_train_test/extracted_data_train_test/",
    classes=['test_imgs2'])
print(len(test_generator.filenames))

train_steps_per_epoch = 22557 // batch_size
#28160  // batch_size
val_steps_per_epoch = 5603 // batch_size
MODEL_FILE = 'simple_cnn_model_final'
filepath = "simple_cnn_adam-{epoch:02d}"
csv_logger = CSVLogger('cnn_log.csv', append = True, separator = ',')
checkpointer = ModelCheckpoint(filepath, monitor = 'acc', verbose = 1, save_best_only = False, save_weights_only = False, 
                               mode = 'max',period = 1)

model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=val_steps_per_epoch,
    callbacks =[csv_logger,checkpointer,learning_rate_reduction])

batch_size = 64
test_generator.reset()
#eval_var= model.evaluate_generator(generator=validation_generator,
#steps=val_steps_per_epoch, verbose = 1)
#STEP_SIZE_TEST=9691//1
test_generator.reset()
pred2=model.predict_generator(test_generator,
steps = 9691/batch_size,
verbose=1)

print(len(pred2))
predicted_class_indices=np.argmax(pred2,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
print(len(filenames))
print(len(predictions))
results3=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results3.to_csv("full_data_results.csv",index=False)
print('All Done!')
results3.head(20)