import numpy as np 
import tensorflow as tf 
import tensorflow_addons as tfa 
import tensorflow_datasets as tfds

class PSO():
    def __init__(self, 
                 TF2_model, 
                 update_w = .99,
                 update_interia = .9,
                 partical_v_limit = .1,
                 update_c1 = 0.,
                 update_c1_lr = 1E-3,
                 update_c2 = 1.,
                 population_size = 20):
        
        self.nnmodel = TF2_model
        self.population_size = population_size
        self.population = self._createPopulation()
        self.update_w = update_w
        self.update_interia = update_interia
        self.update_c1 = update_c1
        self.update_c2 = update_c2
        self.update_c1_lr = update_c1_lr
        self.partical_v_limit = partical_v_limit
        self.pbest = {'fitness':tf.zeros([population_size]), 
                      'weights':tf.zeros(self.population['weights'].shape)}
        self.force_evaluate = True
        #self.SGDOpts = self._createOptimizers(tfa.optimizers.SGDW, learning_rate=1E-4, clipnorm=1.)
        #self.SGDOpts = self._createOptimizers(tfa.optimizers.Yogi, learning_rate=1E-4, clipnorm=1.)
        self.SGDOpts = self._createOptimizers(tf.keras.optimizers.RMSprop, learning_rate=1E-4, clipnorm=1.)
        #self.SGDOpts = self._createOptimizers(tfa.optimizers.RectifiedAdam, learning_rate=1E-4)
        #self.SGDOpts = self._createOptimizers(tf.keras.optimizers.Adamax, learning_rate=1E-4, clipnorm=1.)
        self.SGDopts_g = self._createOptimizers(tf.keras.optimizers.RMSprop, learning_rate=1E-4, clipnorm=1.)
        #self.SGDopts_g = self._createOptimizers(tfa.optimizers.RectifiedAdam, learning_rate=1E-4)
        #self.SGDopts_g = self._createOptimizers(tfa.optimizers.Yogi, learning_rate=1E-4, clipnorm=1.)
        #self.SGDOpts = self._createOptimizers(tfa.optimizers.NovoGrad, learning_rate=1E-4, amsgrad=True, clipnorm=1., weight_decay=1E-4)
        self.SGDOpt = tf.keras.optimizers.SGD(1E-4)
        #self.SGDOpt = tfa.optimizers.SGDW(1E-4, 1E-4)
        #self.SGDOpt = tfa.optimizers.RectifiedAdam(1E-4, clipnorm=1.)
        self.query_models = [tf.keras.models.clone_model(TF2_model) for query_ind in range(self.population_size)] 
    pass 

    def _createPopulation(self):
        self.population = {}
        # creating the weights
        weights = self._flattenWeightsTFKeras()
        #self.population['weights'] = tf.stack([tf.random.normal(weights.shape, stddev=1) for i in range(self.population_size)], axis=0)
        self.population['weights'] = tf.stack([weights * tf.random.normal(weights.shape, stddev=1.) for i in range(self.population_size)], axis=0)
        #self.population['weights'] = tf.stack([tf.random.uniform(weights.shape, minval=-1, maxval=1) for i in range(self.population_size)], axis=0)
        #self.population['velocity'] = tf.stack([tf.random.normal(weights.shape, stddev=0) for i in range(self.population_size)], axis=0)
        self.population['velocity'] = tf.stack([tf.zeros_like(weights) for i in range(self.population_size)], axis=0)
        # creating the nn body for parallel computing
        # self.population['graphs'] = [tf.keras.models.clone_model(self.nnmodel) for i in range(self.population_size)]
        return self.population
    pass 

    def _createOptimizers(self, keras_opt, learning_rate, **kwargs):
        opts = [keras_opt(learning_rate, **kwargs) for i in range(self.population_size)]
        return opts
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

    def updateFitness(self, population, fitness_function, model_loss, SGD = False, batch_dataset = None, g_opt = False):
        fitness_rec = []
        if g_opt :
            SGDOpts = self.SGDOpts
        else :
            SGDOpts = self.SGDopts_g
        pass
        SGD_Optimized_weights = []
        KLD = tf.keras.losses.KLDivergence()
        query_models = [self._recoverFlattenWeightsTFKeras(self.query_models[query_ind], population['weights'][query_ind]) for query_ind in range(self.population_size)]
        #query_models = tf.map_fn(lambda model_idx: self._recoverFlattenWeightsTFKeras(self.query_models[model_idx], population['weights'][model_idx]), elems=tf.range(self.population_size, dtype=tf.int32), parallel_iterations=10)
        for ind_index in range(self.population_size):
            self._recoverFlattenWeightsTFKeras(self.nnmodel, population['weights'][ind_index])
            if SGD: # if the SGD mode is on, the individual will have an SGD update
                if batch_dataset != None: # if provide the dataset, using the self-supervised and crossentropy optimization
                    subject_pred = tf.nn.softmax(self.nnmodel(batch_dataset))
                    def model_and_contrastive_loss():
                        temperature = 5.
                        subject_pred = tf.nn.softmax(self.nnmodel(batch_dataset) / temperature)
                        
                        loss = tf.math.reduce_mean(
                               tf.map_fn(fn=lambda model_idx: KLD(subject_pred, tf.nn.softmax(query_models[model_idx](batch_dataset) / temperature)), elems=tf.range(self.population_size), parallel_iterations=50, fn_output_signature=tf.float32)
                               #tf.map_fn(fn=lambda model_idx: tf.pow(self.nnmodel(batch_dataset) - query_models[model_idx](batch_dataset), 2), elems=tf.range(self.population_size), parallel_iterations=50, fn_output_signature=tf.float32)
                               )
                        
                        #loss = 0
                        #for query_ind in range(self.population_size):
                        #    query_pred = tf.nn.softmax(query_models[query_ind](batch_dataset))
                        #    loss += tf.keras.losses.KLDivergence()(subject_pred, query_pred) + tf.keras.losses.KLDivergence()(query_pred, subject_pred) 
                        #pass
                        return model_loss() + loss 
                    pass
                    SGDOpts[ind_index].minimize(model_and_contrastive_loss, self.nnmodel.trainable_weights)
                else: # if the dataset is not provided, using the crossentropy only
                    #self.SGDOpt.minimize(model_loss, self.nnmodel.trainable_weights) # use single SGD optimizer. this will be infuluence by momentum
                    SGDOpts[ind_index].minimize(model_loss, self.nnmodel.trainable_weights) # use the SGD array for adaptive momentum
                pass
                SGD_Optimized_weights.append(tf.identity(self._flattenWeightsTFKeras()))
            pass
            fitness_rec.append(fitness_function()) # + tf.norm(SGD_Optimized_weights[-1]) * 1E-3)
        pass
        if SGD:
            population['weights'] = tf.stack(SGD_Optimized_weights, axis=0)
        pass
        return tf.concat(fitness_rec, axis=0)
    pass 

    def minimize(self, fitness_function, model_loss, batch_data = None):
        # get fitnesses of each individual
        fitness_rec = self.updateFitness(self.population, fitness_function, model_loss, SGD=True, batch_dataset=batch_data)
        # print(fitness_rec)
        

        if self.force_evaluate:
            self.force_evaluate = False
            self.pbest['fitness'] = tf.identity(fitness_rec)
            self.pbest['weights'] = tf.identity(self.population['weights'])
        else :
            self.pbest['fitness'] = self.updateFitness(self.pbest, fitness_function, model_loss, SGD=True, batch_dataset=batch_data, g_opt=True)
        pass 
        
        # update pbest memory
        rec_cmp = tf.cast(tf.math.less(self.pbest['fitness'], fitness_rec), dtype=tf.float32)
        #rec_cmp = tf.ones_like(rec_cmp)
        #self.pbest['fitness'] = tf.identity(self.pbest['fitness'] * rec_cmp + fitness_rec * (-1 * rec_cmp + 1)) # update the pbest fitness
        #self.pbest['fitness'] = tf.identity(self.pbest['fitness'] * rec_cmp * (1 - self.update_c1_lr) + fitness_rec * (-1 * rec_cmp + 1) * self.update_c1_lr) # update the pbest fitness with momentum strategy
        rec_cmp = tf.reshape(rec_cmp, [-1, 1])
        # print(rec_cmp)
        #self.pbest['weights'] = tf.identity(self.pbest['weights'] * rec_cmp + self.population['weights'] * (-1 * rec_cmp + 1)) # update the pbest weights
        self.pbest['weights'] = tf.identity(self.pbest['weights'] * rec_cmp) + tf.identity(self.pbest['weights'] * (-1 * rec_cmp + 1) * (1 - self.update_c1_lr) + self.population['weights'] * (-1 * rec_cmp + 1) * self.update_c1_lr) # update the pbest fitness with momentum strategy

        # create gbest tensors
        gbest = tf.identity(self.population['weights'][tf.argmin(fitness_rec)])
        #gbest = tf.identity(self.pbest['weights'][tf.argmin(self.pbest['fitness'])])
        gbest_fitness = tf.identity(self.pbest['fitness'][tf.argmin(self.pbest['fitness'])]) 

        # update population 
        # vid+1 = w∙vid+c1∙rand()∙(pid-xid)+c2∙Rand()∙(pgd-xid) 
        # xid+1 = xid+vid
        if np.random.random() > 0.0:
        #if False:
            self.population['velocity'] = tf.clip_by_value(self.population['velocity'], -self.partical_v_limit,  self.partical_v_limit)
            self.population['velocity'] = tf.identity(
                                          self.update_w * tf.identity(self.population['velocity']) +\
                                          #self.update_c1 * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.5, dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                          #self.update_c2 * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.5, dtype=tf.float32)) * tf.identity(gbest - self.population['weights'])
                                          self.update_c1 * tf.math.abs(tf.random.normal([self.population_size, 1], stddev=1., dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                          self.update_c2 * tf.math.abs(tf.random.normal([self.population_size, 1], stddev=1., dtype=tf.float32)) * tf.identity(gbest - self.population['weights'])
                                          #self.update_c1 * tf.math.abs(tf.random.uniform([self.population_size, 1], minval=0, maxval=1, dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                          #self.update_c2 * tf.math.abs(tf.random.uniform([self.population_size, 1], minval=0, maxval=1, dtype=tf.float32)) * tf.identity(gbest - self.population['weights'])
                                          #self.update_c1 * tf.reshape((fitness_rec / self.pbest['fitness']), [-1, 1]) * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.5, dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                          #self.update_c2 * tf.reshape((fitness_rec / gbest_fitness), [-1, 1]) * tf.math.abs(tf.random.normal(self.population['velocity'].shape, stddev=.5, dtype=tf.float32)) * tf.identity(gbest - self.population['weights'])
                                          #self.update_c1 * tf.reshape((fitness_rec / self.pbest['fitness']), [-1, 1]) * tf.math.abs(tf.random.normal([self.population_size, 1], stddev=.5, dtype=tf.float32)) * tf.identity(self.pbest['weights'] - self.population['weights']) +\
                                          #self.update_c2 * tf.reshape((fitness_rec / gbest_fitness), [-1, 1]) * tf.math.abs(tf.random.normal([self.population_size, 1], stddev=.5, dtype=tf.float32)) * tf.identity(gbest - self.population['weights']) 
                                          )
            self.population['weights'] = tf.identity(self.population['velocity'] + self.population['weights'])

            # update w 
            self.update_w *= self.update_interia
        pass
        
        return self.pbest['fitness'][tf.argmin(self.pbest['fitness'])]
    pass 

    def getTopModel(self, fitness_function):
        # get fitnesses of each individual
        fitness_rec = self.updateFitness(self.pbest, fitness_function, True)

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
    mnist_tr_iter = iter(mnist_tr.batch(512).repeat())

    cnn = sample_cnn()
    data_fetcher = mnist_tr_iter.next()
  
    ###################
    def loss():
        #data_fetcher = mnist_tr_iter.next()
        #print(cnn(data_fetcher['image']))
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(np.reshape(data_fetcher['label'], [-1, 1]), 10, axis=-1), logits = cnn(data_fetcher['image'])))
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(data_fetcher['label'], 10, axis=-1), cnn(data_fetcher['image']), from_logits=True))
    pass   
    ###################

    print(loss())

    opt = PSO(cnn)
    
    print("PSO optimization ...")
    # PSO optimization
    # while(1):
    for steps in range(1000):
        data_fetcher = mnist_tr_iter.next()
        print(opt.minimize(loss))
    pass
    opt.getTopModel(loss)

    #print("SGD optimization ...")
    # opt = tfa.optimizers.NovoGrad(1E-4)
    #opt = tf.keras.optimizers.RMSprop(1E-4)
    #for steps in range(1000):
    #    data_fetcher = mnist_tr_iter.next()
    #    opt.minimize(loss, var_list = cnn.trainable_weights)
    #    print(loss())
    #pass

pass 

if __name__ == "__main__":
   main() 
pass
