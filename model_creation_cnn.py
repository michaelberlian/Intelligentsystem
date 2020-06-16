import numpy as np 
import matplotlib.pyplot as plt
import random 
import tensorflow as tf

#setup keras
tf.keras.backend.set_floatx('float64')

# setup the datasets
X = np.load("X.npy")
Y = np.load("Y.npy")

X = np.array(X)
Y = np.array(Y)

X = X/255

x_train = X
y_train = Y


conv = [1,2]
dense = [0,1,2]

#train different models
for i in range (1,3) :
    
    NAME = 'tfboard-{}-conv-{}-dense'.format(0,i)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs_final/{}'.format(NAME))

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())

    for r in range (i) :
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(26))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size = 32, epochs=10, validation_split=0.1,callbacks = [tensorboard])

    name = '{}-conv-{}-dense-2'.format(0,i)
    model.save(name)

for i in conv:
    for p in dense :

        NAME = 'tfboard-{}-conv-{}-dense'.format(i,p)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs_final/{}'.format(NAME))
        model = tf.keras.models.Sequential()

        for q in conv :
            model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = x_train.shape[1:]))
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

        model.add(tf.keras.layers.Flatten())

        for r in dense :
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Activation("relu"))

        model.add(tf.keras.layers.Dense(26))
        model.add(tf.keras.layers.Activation('softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size = 32, epochs=5, validation_split=0.1, callbacks = [tensorboard])

        name = '{}-conv-{}-dense-2'.format(i,p)
        model.save(name)