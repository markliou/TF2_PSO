# TF2_PSO
Use evolutionary algorithm to optimize the weights in the neural network made from Keras 

# Prerequest
1. Tensorflow-2.3 (or above)
2. Tensorflow-addons (option. You can delete it as you want)
3. Tensorflow-datasets (option. You can delete it as you want)

# Features
* The PSO is implemented with TF-Keras and python. The APIs are also try to make as smilar as TF-Keras API.
* The hybrid optimization is also accesseble using this algorithm 

# Example
The sample code can be find in the 'main.py'.

## The API
```python
import TF2_PSO_BP as PSO

def cnn()
    def sample_cnn():
    Input = tf.keras.Input([28,28,1])
    nInput = Input/128 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(nInput) #[14,14]
    conv2 = tf.keras.layers.Conv2D(64, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(128, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(10, activation=None)(fc3)

    return tf.keras.Model(inputs=Input, outputs=out)
pass 

sample_cnn = cnn() # make the cnn
opt = PSO.PSO(cnn) # initiate the PSO object
```

Passing neural network object in to the optimizer is due to the PSO will make the population 
according to the neural network object. 
To make the optimization process:
```python
(opt.minimize(loss)
```