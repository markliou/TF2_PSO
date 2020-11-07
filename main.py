import tensorflow as tf 
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import TF2_PSO_BP as PSO

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


def main():
    mnist = tfds.load('MNIST')
    mnist_tr, mnist_ts = mnist['train'], mnist['test']
    mnist_tr_iter = iter(mnist_tr.batch(128).repeat())

    cnn = sample_cnn()
    data_fetcher = mnist_tr_iter.next()
  
    ###################
    def loss():
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(data_fetcher['label'], 10, axis=-1), cnn(data_fetcher['image']), from_logits=True))
    pass   
    ###################

    opt = PSO.PSO(cnn)
    
    print("PSO optimization ...")
    for steps in range(10000000):
        data_fetcher = mnist_tr_iter.next()
        print(opt.minimize(loss))
    pass
    opt.getTopModel(loss)

    print("SGD optimization ...")
    opt = tfa.optimizers.NovoGrad(1E-4)
    for steps in range(1000):
       data_fetcher = mnist_tr_iter.next()
       opt.minimize(loss, var_list = cnn.trainable_weights)
       print(loss())
    pass

pass 

if __name__ == "__main__":
   main() 
pass
