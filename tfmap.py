import tensorflow as tf 
import numpy as np

def sample_nn():
    Input = tf.keras.Input([3])
    fc1 = tf.keras.layers.Dense(3)(Input)
    fc2 = tf.keras.layers.Dense(2)(fc1)
    out = tf.keras.layers.Dense(1)(fc2)
    return tf.keras.Model(inputs=Input, outputs=out)

def assignWeights(nn, wights):
    pass


nn = sample_nn()
print(nn(np.random.random([2,3])))

print(nn.trainable_weights)
