from tensorflow.keras import preprocessing,callbacks
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
import numpy as np


import matplotlib.pyplot as plt


def my_cnn_task_1() :
    """ Trains and evaluates CNN image classifier on the sea animals dataset.
        Saves the trained model. """
    # load datasets
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    test_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    print("Classes:", training_set.class_names)

    # build the model
    m = Sequential()
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    m.add(Rescaling(1/255))
    m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation='softmax'))

    # setting and training
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 100
    print("Training.")
    history  = m.fit(training_set, batch_size=32, epochs=epochs,verbose=1, callbacks = [callback])
    # print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])

    # testing
    print("Testing.")
    score = m.evaluate(test_set, verbose=0)
    print('Test accuracy:', score[1])

    # saving the model
    print("Saving the model in my_cnn.h5.")
    m.save("my_cnn.h5")


def my_cnn_task_2() :
    """ Trains and evaluates CNN image classifier on the sea animals dataset.
        Saves the trained model. """
    # load datasets
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    print("Classes:", training_set.class_names)

    # build the model
    m = Sequential()
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    m.add(Rescaling(1/255))
    m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.1))
    m.add(Conv2D(128, kernel_size=(4, 4), activation='relu'))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Dropout(0.2))
    m.add(Conv2D(128, kernel_size=(4, 4), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.3))
    m.add(Flatten())
    m.add(Dense(64, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation='softmax'))

    # setting and training
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 100
    print("Training.")
    history  = m.fit(training_set, batch_size=32, epochs=epochs,verbose=1, callbacks = [callback])
    # print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])

    # saving the model
    print("Saving the model in my_cnn2.h5.")
    m.save("my_cnn2.h5")


def fine_tune() :
    """ Trains and evaluates CNN image classifier on the sea animalss dataset.
        Saves the trained model. """
    # load datasets
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))

    print("Classes:", training_set.class_names)

    # Load a general pre-trained model.
    base_model = VGG16(weights='imagenet', include_top=False)

    x = base_model.output # output layer of the base model

    x = GlobalAveragePooling2D()(x)
    # a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    output_layer = Dense(5, activation='softmax')(x)

    # this is the model we will train
    m = Model(inputs=base_model.input, outputs=output_layer)

    # train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional base model layers
    for layer in base_model.layers:
        layer.trainable = False

    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 100
    print("Training.")
    history = m.fit(training_set, batch_size=32, epochs=epochs,verbose=1, callbacks = [callback])
    print(history.history["accuracy"])

    # saving the model
    print("Saving the model in my_fine_tuned.h5.")
    m.save("my_fine_tuned.h5")

def test_internal(m):
    # testing
    test_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                          validation_split=0.2,
                                                          subset="validation",
                                                          label_mode="categorical",
                                                          seed=0,
                                                          image_size=(100, 100))
    print("Testing.")
    score = m.evaluate(test_set, verbose=0)
    print('Test accuracy:', score[1])


def error_analysis(image, m):
    img = preprocessing.image.load_img(image, target_size=(100, 100))
    img_arr = preprocessing.image.img_to_array(img)
    img_cl = img_arr.reshape(1, 100, 100, 3)
    score = m.predict(img_cl)
    predicted_class = np.argmax(score)
    print(predicted_class)

if __name__ == '__main__':
    # my_cnn_task_2()
    m1 = load_model('my_cnn2.h5')
    # test_internal(m1)
    m1.summary()
    # test_internal(m1)
    # m2 = load_model('my_fine_tuned.h5')
    # print('otter')
    # for i in range(1,6):
    #     print(f'iteration: {i}')
    #     print('my_cnn2')
    #     error_analysis(f'test_animals/otter/o{i}.jpg', m1)
    #     print("vgg")
    #     error_analysis(f'test_animals/otter/o{i}.jpg', m2)
    #
    # print('turtle')
    # for i in range(1,6):
    #     print(f'iteration: {i}')
    #     print('my_cnn2')
    #     error_analysis(f'test_animals/turtle/t{i}.jpg', m1)
    #     print("vgg")
    #     error_analysis(f'test_animals/turtle/t{i}.jpg', m2)
    #
    # print('sea urchin')
    # for i in range(1,6):
    #     print(f'iteration: {i}')
    #     print('my_cnn2')
    #     error_analysis(f'test_animals/sea_urchin/su{i}.jpg', m1)
    #     print("vgg")
    #     error_analysis(f'test_animals/sea_urchin/su{i}.jpg', m2)
    #
    # print('sharks')
    # for i in range(1,6):
    #     print(f'iteration: {i}')
    #     print('my_cnn2')
    #     error_analysis(f'test_animals/sharks/sh{i}.jpg', m1)
    #     print("vgg")
    #     error_analysis(f'test_animals/sharks/sh{i}.jpg', m2)

    # test_internal(m)
    # m.summary()
    # my_cnn_task_2()
    # m = load_model('my_cnn2.h5')
    # test_internal(m)
    # m.summary()
    # fine_tune()
    # m = load_model('my_fine_tuned.h5')
    # test_internal(m)
    # m.summary()
