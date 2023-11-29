import processing
import os
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels, _) = processing.create_data("cifar-10-batches-py", "data")
(test_images, test_labels, _) = processing.create_data("cifar-10-batches-py", "test")

#normalize
train_images = train_images / 255
test_images = test_images / 255

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)


