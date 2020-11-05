import numpy as np 
import tensorflow as tf 
import tensorflow_addons as tfa 
import tensorflow_datasets as tfds

class PSO():
    def __init__(self, 
                 TF2_model, 
                 update_w = .2,
                 update_c1 = 2,
                 update_c2 = 2,
                 population_size = 100):
        self.nnmodel = TF2_model
        self.population_size = population_size
        self.population = self._createPopulation()
        self.update_w = update_w
        self.update_c1 = update_c1
        self.update_c2 = update_c2
        self.pbest = {'fitness':tf.zeros([population_size]), 
                      'weights':tf.zeros(self.population['weights'].shape)}
        self.force_evaluate = True
    pass 

    def _createPopulation(self):
        self.population = {}
        # creating the weights
        weights = self._flattenWeightsTFKeras()
        self.population['weights'] = tf.stack([tf.random.normal(weights.shape, stddev=.5) for i in range(self.population_size)], axis=0)
        self.population['velocity'] = tf.stack([tf.random.normal(weights.shape, stddev=.05) for i in range(self.population_size)], axis=0)
        # creating the nn body for parallel computing
        # self.population['graphs'] = [tf.keras.models.clone_model(self.nnmodel) for i in range(self.population_size)]
        return self.population
    pass 

    def _flattenWeightsTFKeras(self):
        flated_weights = tf.Variable(tf.concat([tf.reshape(weights, [-1]) for weights in self.nnmodel.trainable_weights], axis=-1))
        return flated_weights
    pass 

    def _recoverFlattenWeightsTFKeras(self, model, flated_weights):
        access_index = 0
        for model_tesnsor in model.trainable_weights:
            element_shape = model_tesnsor.shape.as_list()
            element_number = np.ones(element_shape).sum().astype('int')
            model_tesnsor.assign(tf.reshape(flated_weights[access_index:access_index + element_number], element_shape))
            access_index += element_number
        pass
        return model
    pass 

    def update_fitness(self, fitness_function):
        fitness_rec = []
        for ind_index in range(self.population_size):
            self._recoverFlattenWeightsTFKeras(self.nnmodel, self.population['weights'][ind_index])
            fitness_rec.append(fitness_function())
        pass 
        return tf.concat(fitness_rec, axis=0)
    pass 

    def minimize(self, fitness_function):
        # get fitnesses of each individual
        fitness_rec = self.update_fitness(fitness_function)
        # print(fitness_rec)

        # take change to jump out of the outlier
        # 有可能當下抽樣計算出的loss異常的高。如果抽到這樣的outliner可能會把族群trap住。
        # 所以pbest的紀錄用moving average稍微處理
        if self.force_evaluate:
            self.force_evaluate = False
            self.pbest['fitness'] = tf.identity(fitness_rec)
            self.pbest['weights'] = tf.identity(self.population['weights'])
        else :
            self.pbest['fitness'] = tf.identity(fitness_rec + self.pbest['fitness'])/2 
        pass 
        
        # update pbest memory
        rec_cmp = tf.cast(tf.math.less(self.pbest['fitness'], fitness_rec), dtype=tf.float32)
        self.pbest['fitness'] = tf.identity(self.pbest['fitness'] * rec_cmp + fitness_rec * (-1 * rec_cmp + 1)) # update the pbest fitness
        rec_cmp = tf.reshape(rec_cmp, [-1, 1])
        # print(rec_cmp)
        self.pbest['weights'] = tf.identity(self.pbest['weights'] * rec_cmp + self.population['weights'] * (-1 * rec_cmp + 1)) # update the pbest weights
        
        # create gbest tensors
        gbest = tf.identity(self.population['weights'][tf.argmin(fitness_rec)])

        # update population 
        # vid+1 = w∙vid+c1∙rand()∙(pid-xid)+c2∙Rand()∙(pgd-xid) 
        # xid+1 = xid+vid
        self.population['velocity'] = tf.identity(
                                      self.update_w * tf.identity(self.population['velocity']) +\
                                      self.update_c1 * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.05, dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                      self.update_c2 * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.05, dtype=tf.float32)) * tf.identity(gbest - self.population['weights'])
                                      )
        self.population['weights'] = tf.identity(self.population['velocity'] + self.population['weights'])

        return self.pbest['fitness'][tf.argmin(self.pbest['fitness'])]
    pass 

    def getTopModel(self):
        # get fitnesses of each individual
        fitness_rec = self.update_fitness()

        # create gbest tensors
        selected_ind = self.pbest['weights'][tf.argmax(fitness_rec)] 

        # recover the wieghts to cnn
        self._recoverFlattenWeightsTFKeras(self.nnmodel, selected_ind)
        return self.nnmodel
    pass

pass 

def sample_cnn():
    Input = tf.keras.Input([28,28,1])
    nInput = Input/128 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(nInput) #[14,14]
    conv2 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(10, activation=None)(fc3)

    return tf.keras.Model(inputs=Input, outputs=out)
pass 


def main():
    mnist = tfds.load('MNIST')
    mnist_tr, mnist_ts = mnist['train'], mnist['test']
    mnist_tr_iter = iter(mnist_tr.batch(32).repeat())

    cnn = sample_cnn()
    data_fetcher = mnist_tr_iter.next()
  
    ###################
    def loss():
        #print(cnn(data_fetcher['image']))

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = np.reshape(data_fetcher['label'], [-1, 1]), logits = cnn(data_fetcher['image'])))
    pass   
    ###################

    print(loss())

    opt = PSO(cnn)
    print(opt.minimize(loss))
    
    # optimization
    while(1):
    # for steps in range(20):
        # data_fetcher = mnist_tr_iter.next()
        print(opt.minimize(loss))
    pass


pass 

if __name__ == "__main__":
   main() 
pass