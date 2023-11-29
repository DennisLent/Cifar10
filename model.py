import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import processing
import os


def create_model(ind):
    """
    function to create a model from an individuals parameters
    """

    model = models.Sequential()
    (conv1, conv2, conv3, s_conv1, s_conv2, s_conv3, s_fc1, s_fc2) = ind.genes

    #create the model
    model.add(Conv2D(conv1, (s_conv1, s_conv1), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(conv2, (s_conv2, s_conv2), activation='relu'))
    model.add(Conv2D(conv3, (s_conv3, s_conv3), activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(s_fc1, activation='relu'))
    model.add(Dense(s_fc2, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def sim_model(model, train, test, epochs):
    """
    function to create get the testing accuracy of the model
    """
    train_images, train_labels = train
    test_images, test_labels = test
    history = model.fit(train_images, train_labels, epochs=epochs, verbose=1, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    return test_acc
    



""" model = models.Sequential()

# Convolutional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Fully connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.summary()
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}") """
