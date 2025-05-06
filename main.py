import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist # Load the MNIST dataset of handwritten digits

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Split the dataset into training and testing sets
x_train = x_train / 255.0 # Normalize the training data to the range [0, 1]



