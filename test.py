import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# import tensorflow.compat.v1 as tf
# vars = tf.get_collection(tf.GraphKeys.TARGETS)
# print(tf.GraphKeys.TARGETS)
# print(vars)
# exit()

TRAIN_DIR  = '/home/dionbudi/Documents/riset/cnn/train/'
TEST_DIR = '/home/dionbudi/Documents/riset/cnn/test/'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'mnist-convnet'

# 0-4 aja
def create_label(image_name):
    word_label = image_name.split('/')[7]
    if word_label == '0':
        return np.array([1,0,0,0,0])
    elif word_label == '1':
        return np.array([0,1,0,0,0])
    elif word_label == '2':
        return np.array([0,0,1,0,0])
    elif word_label == '3':
        return np.array([0,0,0,1,0])
    elif word_label == '4':
        return np.array([0,0,0,0,1])

def create_train_data():
    training_data = []
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        for img in tqdm(os.listdir(TRAIN_DIR+'/'+folder)):
            path = os.path.join(TRAIN_DIR+folder, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label(path)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for folder in tqdm(os.listdir(TEST_DIR)):
        for img in tqdm(os.listdir(TEST_DIR+'/'+folder)):
            path = os.path.join(TEST_DIR+folder, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img_data), create_label(path)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data = create_test_data()

train = train_data[:-500]
test = test_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]


# model
# tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='output')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit(X_train, y_train, n_epoch=10, validation_set=(X_test, y_test), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

fig = plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    # print(model_out)
    # print(img_num)
    # print(np.argmax(img_num))

    if np.argmax(model_out) == 0:
        str_label = '0'
    elif np.argmax(model_out) == 1:
        str_label = '1'
    elif np.argmax(model_out) == 2:
        str_label = '2'
    elif np.argmax(model_out) == 3:
        str_label = '3'
    elif np.argmax(model_out) == 4:
        str_label = '4'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()



# print(convnet)
# exit()

# print(create_label('/home/dionbudi/Documents/riset/cnn/train/4/59975.png'))