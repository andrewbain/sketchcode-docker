import os
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

import numpy as np
from config import Config

class SketchModel(object):
    def __init__(self, Config):
        self.NB_EPOCHS = 80
        self.BAT_SIZE = 32
        self.FC_SIZE = 1024
        self.image_width = 224
        self.image_height = 224
        self.data = []
        self.labels = []
        self.config = Config()

    def process_image_for_vgg(self, image_path):
        image = load_img(image_path, target_size=(self.image_width, self.image_height))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = vgg_preprocess_input(image)

        return image

    def process_image_for_inception(self, image_path):
        image = load_img(image_path, target_size=(299,299))
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return image

    def process_image_for_custom(self, image_path):
        image = load_img(image_path, grayscale=True, target_size=(200, 200))

        # convert the image pixels to a numpy array
        image = img_to_array(image)
        return image


    def prepare_test_train_data(self, model_name):
        classes = self.config.load_classes()

        for c in enumerate(classes):
            for filename  in os.listdir('images/' + c[1]['directory']):
                if filename != '.DS_Store':
                    if model_name == 'vgg16':
                        image = self.process_image_for_vgg('images/{}/{}'.format(c[1]['directory'],filename))
                    if model_name == 'inception':
                        image = self.process_image_for_inception('images/{}/{}'.format(c[1]['directory'], filename))
                    if model_name == 'custom':
                        image = self.process_image_for_custom('images/{}/{}'.format(c[1]['directory'], filename))
                    self.data.append(image)
                    self.labels.append(c[1]['id'])
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        self.data = np.array(self.data, dtype="float")
        self.labels = np.array(self.labels)

        (self.trainX, self.testX, trainY, testY) = train_test_split(self.data, self.labels, test_size=0.25, random_state=42)

        # convert the labels from integers to vectors
        self.trainY = to_categorical(trainY, num_classes=self.config.num_classes)
        self.testY = to_categorical(testY, num_classes=self.config.num_classes)

    def add_new_last_layer(self, baseModel, num_classes):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.FC_SIZE, activation='relu')(x)  # new FC layer, random init
        predictions = Dense(num_classes, activation='softmax')(x)  # new softmax layer
        newModel = Model( inputs=baseModel.inputs, outputs=predictions )
        return newModel

    def trainVGG(self):
        self.prepare_test_train_data('vgg16')
        aug = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest")
        baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(self.image_width, self.image_height, 3))
        for layer in baseModel.layers[:5]:
            layer.trainable = False
        model = self.add_new_last_layer(baseModel, self.config.num_classes)

        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        aug.fit(self.trainX)
        H = model.fit_generator(aug.flow(self.trainX, self.trainY, batch_size=self.BAT_SIZE),
                                validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // self.BAT_SIZE,
                                epochs=self.NB_EPOCHS, verbose=1)

        model.save('my-vgg-model.h5')


    def trainInception(self):

        self.prepare_test_train_data('inception')
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.config.num_classes, activation='sigmoid')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode="nearest")
        aug.fit(self.trainX)
        # train the model on the new data for a few epochs
        model.fit_generator(aug.flow(self.trainX, self.trainY, batch_size=self.BAT_SIZE),
                                validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // self.BAT_SIZE,
                                epochs=3, verbose=1)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:6]:
            layer.trainable = False
        for layer in model.layers[6:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        h = model.fit_generator(aug.flow(self.trainX, self.trainY, batch_size=self.BAT_SIZE),
                                validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // self.BAT_SIZE,
                                epochs=20, verbose=1)
        model.save('my-inception-model.h5')
        # self.plot_training(h)

    # def plot_training(self, H):
    #     # plot the training loss and accuracy
    #     plt.style.use("ggplot")
    #     plt.figure()
    #     N = self.NB_EPOCHS
    #     print(H.history)
    #     plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    #     plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    #     plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    #     plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    #     plt.title("Training Loss for Sketch Model")
    #     plt.xlabel("Epoch #")
    #     plt.ylabel("Loss/Accuracy")
    #     plt.legend(loc="lower left")
    #     plt.savefig('results.pdf')

    def predict(self, image_path):
        image = self.process_image_for_custom(image_path)
        image = image.reshape(1, 200, 200, 1).astype('float32')
        image = image / 255
        
        yhat = self.config.model.predict_classes(image)
        markup = self.get_markup(yhat[0])
        return markup

    def get_markup(self, id):
        return [c for c in self.config.load_classes() if c['id'] == id][0]



    def custom_model(self):
        # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(200, 200, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def train_custom_model(self):
        self.prepare_test_train_data('custom')
        # reshape to be [samples][pixels][width][height]

        self.trainX = self.trainX.reshape(self.trainX.shape[0], 200, 200, 1).astype('float32')
        self.testX = self.testX.reshape(self.testX.shape[0], 200, 200, 1).astype('float32')
        # normalize inputs from 0-255 to 0-1
        self.trainX = self.trainX / 255
        self.testX = self.testX / 255
        # one hot encode outputs

        self.num_classes = self.testY.shape[1]

        model = self.custom_model()
        aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode="nearest")
        aug.fit(self.trainX)
        # Fit the model
        h = model.fit_generator(aug.flow(self.trainX, self.trainY, batch_size=32), validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // 32, epochs=self.NB_EPOCHS)
        # Final evaluation of the model
        scores = model.evaluate(self.testX, self.testY, verbose=0)
        model.save('./models/custom-model-latest.h5')
        # self.plot_training(h)
        print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

# SketchModel(Config).train_custom_model()


