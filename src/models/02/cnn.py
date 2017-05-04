from ... import utils

import csv, time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

dataset = utils.get_dataset(filetype='.mat')
(train, valid, test) = utils.load_data(dataset)

data_augmentation = True
num_classes = 3

model = Sequential()

model.add(ZeroPadding2D(padding=(1,1), input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(1024, (3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(1024, (1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(1024, (3,3)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

from keras.utils.layer_utils import print_summary
print_summary(model)

epochs = 50
lrate = 0.01
decay = lrate/epochs
batch_size=64
sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    res = model.fit(train['x'], train['y'],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(valid['x'], valid['y']),
                    shuffle=True,
                    verbose=2)
    score = model.evaluate(train['x'], train['y'])
else:
    print('Using data augmentation.')
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zca_whitening=True,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(zca_whitening=True)
    test_datagen.fit(valid['x'])
    validation_generator = test_datagen.flow(valid['x'], valid['y'])

    train_datagen.fit(train['x'])
    train_generator = train_datagen.flow(train['x'], train['y'], batch_size=batch_size)

    res = model.fit_generator(train_generator,
                              steps_per_epoch=train['x'].shape[0] // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=400,
                              shuffle=True,
                              verbose=2)

test['y_pred'] = model.predict_classes(test['x'], verbose=0)
print(test['y_pred'])

with open('model.out.csv','wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    for pred in test['y_pred']:
        wr.writerow([pred])

utils.plot_model_history(res)