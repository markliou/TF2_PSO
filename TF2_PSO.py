import numpy as np 
import tensorflow as tf 
import tensorflow_addons as tfa 

class PSO():
    def __init__(self, 
                 TF2_model, 
                 population_size=50, 
                 iteration_no=None):
        self.nnmodel = TF2_model
        self.population_size = population_size
        self.iteration_no = iteration_no
        self.population = self._createPopulation()
    pass 

    def _createPopulation(self):
        weights = self._flattenWeightsTFKeras()
        self.population = tf.stack([tf.random.uniform(weights.shape) for i in range(self.population_size)], axis=0)
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
            model_tesnsor.assign(tf.reshape(flated_weights[access_index:access_index+element_number], element_shape))
            access_index += element_number
        pass
        return model
    pass 

    def minimize(self):
        if self.iteration_no:
            self.optimizationProcess(dataset_iter)
        else:
            self.optimizationProcess_OneStep(dataset)
        pass 
    pass 

    def optimizationProcess(self, dataset_iter):
        
        pass 

    def optimizationProcess_OneStep(self, dataset):
        
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
    out = tf.keras.layers.Dense(1, activation=None)(fc3)

    return tf.keras.Model(inputs=Input, outputs=out)
pass 


def main():
    cnn = sample_cnn()
pass 

if __name__ == "__main__":
   main() 
pass