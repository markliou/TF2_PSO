import tensorflow as tf 
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import TF2_PSO_BP as PSO

def sample_cnn():
    Input = tf.keras.Input([32, 32, 3])
    nInput = Input/128 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(nInput) #[14,14]
    conv1_1 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv1)
    conv1_2 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv1_1)
    conv1_3 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv1_2)
    conv2 = tf.keras.layers.Conv2D(64, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv1_3) #[7,7]
    conv2_1 = tf.keras.layers.Conv2D(128, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv2)
    conv2_2 = tf.keras.layers.Conv2D(128, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv2_1)
    conv3 = tf.keras.layers.Conv2D(128, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv2_2) #[4,4]
    conv3_1 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv3)
    conv3_2 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_initializer="he_uniform")(conv3_1)
    fc = tf.keras.layers.Flatten()(conv3_2)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer="he_uniform")(fc)
    fc2 = tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer="he_uniform")(fc1)
    fc3 = tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer="he_uniform")(fc2)
    out = tf.keras.layers.Dense(365, activation=None)(fc3)

    return tf.keras.Model(inputs=Input, outputs=out)
pass 


def main():
    #mnist = tfds.load('MNIST')
    #mnist = tfds.load('FashionMNIST')
    #mnist = tfds.load('KMNIST')
    #mnist = tfds.load('Cifar10')
    #mnist = tfds.load('SvhnCropped')
    #mnist_tr, mnist_ts = mnist['train'], mnist['test']
    #mnist_tr_iter = iter(mnist_tr.batch(256).repeat())
    #mnist_ts_iter = iter(mnist_tr.batch(10000))

    place365 = tfds.load('Places365Small')
    place365_tr, place365_ts = place365['train'], place365['test']
    place365_tr_iter = iter(place365_tr.batch(512).repeat())
    place365_ts_iter = iter(place365_tr.batch(1000))

    cnn = sample_cnn()
    #data_fetcher = mnist_tr_iter.next()
    data_fetcher = place365_tr_iter.next()
    #mnist_ts_ds = mnist_ts_iter.next()
    mnist_ts_ds = place365_ts_iter.next()
  
    ###################
    def loss_bp():
        #return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(data_fetcher['label'], 365, axis=-1), cnn(data_fetcher['image']), from_logits=True, label_smoothing=.1))
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(data_fetcher['label'], 365, axis=-1), cnn(tf.image.resize(data_fetcher['image'], [32, 32])), from_logits=True, label_smoothing=.1))
    pass   

    def loss():
        #return tf.identity(1 - (tf.reduce_mean(tf.cast(tf.math.equal(data_fetcher['label'], tf.argmax(cnn.predict(data_fetcher['image']), axis=-1)), dtype=tf.float32))))
        return tf.identity(1 - (tf.reduce_mean(tf.cast(tf.math.equal(data_fetcher['label'], tf.argmax(cnn.predict(tf.image.resize(data_fetcher['image'], [32, 32])), axis=-1)), dtype=tf.float32)))) # accuracy
    pass
    ###################

    opt = PSO.PSO(cnn)
    print("PSO optimization ...")
    for steps in range(1000000):
        #data_fetcher = mnist_tr_iter.next()
        data_fetcher = place365_tr_iter.next()
        loss_1 = opt.minimize(loss, loss_bp, tf.image.resize(data_fetcher['image'], [32, 32]))
        loss_2 = loss_bp()
        #pred = tf.reduce_mean(tf.cast(tf.math.equal(mnist_ts_ds['label'], tf.argmax(cnn.predict(mnist_ts_ds['image']), axis=-1)), dtype=tf.float32))
        pred = tf.reduce_mean(tf.cast(tf.math.equal(mnist_ts_ds['label'], tf.argmax(cnn.predict(tf.image.resize(mnist_ts_ds['image'], [32, 32])), axis=-1)), dtype=tf.float32))
        print("step:{} loss_1:{:.5f} loss_2:{:.5f} ts_aac:{:.2f}".format(steps, loss_1, loss_2, pred))
    pass
    opt.getTopModel(loss)
    exit()

    print("SGD optimization ...")
    #opt = tfa.optimizers.NovoGrad(1E-4)
    #opt = tf.keras.optimizers.RMSprop(1E-4, clipnorm=1.)
    opt = tfa.optimizers.Yogi(1E-4, clipnorm=1.)
    for steps in range(200000000):
        #data_fetcher = mnist_tr_iter.next()
        data_fetcher = place365_tr_iter.next()
        opt.minimize(loss_bp, var_list = cnn.trainable_weights)
        loss_1 = loss()
        loss_2 = loss_bp()
        #pred = tf.reduce_mean(tf.cast(tf.math.equal(mnist_ts_ds['label'], tf.argmax(cnn.predict(mnist_ts_ds['image']), axis=-1)), dtype=tf.float32))
        pred = tf.reduce_mean(tf.cast(tf.math.equal(mnist_ts_ds['label'], tf.argmax(cnn.predict(tf.image.resize(mnist_ts_ds['image'], [32, 32])), axis=-1)), dtype=tf.float32))
        if steps % 50 == 0:
            print("step:{} loss_1:{:.5f} loss_2:{:.5f} ts_aac:{:.2f}".format(steps, loss_1, loss_2, pred))
        pass
    pass

pass 

if __name__ == "__main__":
   main() 
pass
