import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from copy import deepcopy
from typing import Any
import tensorflow as tf
import numpy as np
import pickle
import threading
import keras
import time

from Logger import *
from Random import *


class Model:

    '''
    Base class for our ML models.
    '''

    def __init__(
        self,
        logger: Logger,
        random: Random,
        sync: bool
    ) -> None:
        
        self.logger = logger
        self.random = random
        self.tqdm = lambda x: x

        '''
        Any models and fims shared by other devices are stored
        in a queue to be merged when needed.
        '''

        self.model: keras.models.Model
        self.fim: list = []

        self.model_queue: dict [int, dict [str, list]] = {}
        self.received_weights: dict [str, list] = {}
        self.received_fims: dict [str, list] = {}
        self.lock: threading.Lock = threading.Lock()

        '''
        Data from training and evaluating.
        '''

        self.x_train: list [np.ndarray]
        self.y_train: list [np.ndarray]
        self.x_test: list [np.ndarray]
        self.y_test: list [np.ndarray]

        self.train_ds: tf.data.Dataset
        self.test_ds: tf.data.Dataset

        '''
        Results of the training.
        '''

        self.train_acc: list [float] = []
        self.train_loss: list [float] = []
        self.test_acc: list [float] = []
        self.test_loss: list [float] = []

        '''
        In sync mode we don't actively merge.
        In async mode we check after each mini-batch update
        for any received weights and merge them.
        '''

        self.sync = sync
        self.interrupt = False
        self.interrupt_lock = threading.Lock()

        '''
        Hyperparameters.
        '''

        self.optimizer: keras.optimizers.Optimizer
        self.loss_fn: keras.losses.Loss
        self.metric_fn: keras.metrics.Metric

        self.learning_rate: float
        self.l2_term: float
        self.prox_term: float
        self.curv_term: float
    
    def create(
        self,
        input_shape,
        num_outputs,
        learning_rate,
        l2_term,
        prox_term,
        curv_term,
        *args
    ) -> None:
        
        self.learning_rate = learning_rate
        self.l2_term = l2_term
        self.prox_term = prox_term
        self.curv_term = curv_term

        self.optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            clipnorm=1.0
        )

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        self.metric_fn = keras.metrics.SparseCategoricalAccuracy()

    def preheat(
        self,
        train_files: list [str],
        test_files: list [str],
        label_distribution: tuple [float, float],
        sample_distribution: tuple [float, float]
    ) -> None:

        '''
        We load the dataset for training and evaluation from the storage.
        And, create a non-IID distribution if need by specifying a 
        subset of data, which is split by class/label in different files.
        '''

        ls, le = label_distribution
        ss, se = sample_distribution

        train_files = train_files[ls:le] + (
            [] if le <= len(train_files) 
            else train_files[:le - len(train_files)]
        )

        if self.logger: self.logger.log(
            "Model.preheat",
            "Preheating files: \n| [%s][%.2f:%.2f] \n| [%s]",
            ", ".join(train_files), *sample_distribution,
            ", ".join(test_files)
        )

        '''
        Load training data.
        '''

        self.x_train, self.y_train = [], []

        for filename in train_files:
            
            '''
            Read a subset of samples from the storage.
            '''

            try:

                with open(filename, 'rb') as file:

                    data = pickle.load(file)
                    n = data[0].shape[0]
                    
                    if se > 1:

                        self.x_train.append(
                            np.vstack((
                                data[1][int(ss * n): int(se * n)],
                                data[1][: int((se - 1) * n)]
                            ))
                        )
 
                        self.y_train.append(
                            np.hstack((
                                data[0][int(ss * n): int(se * n)],
                                data[0][: int((se - 1) * n)]
                            ))
                        )

                    else:

                        self.x_train.append(
                            data[1][int(ss * n): int(se * n)]
                        )

                        self.y_train.append(
                            data[0][int(ss * n): int(se * n)]
                        )

            except Exception as e: print(e)

        '''
        Load evaluation data.
        '''

        self.x_test, self.y_test = [], []

        for filename in test_files:

            try:

                with open(filename, 'rb') as file:

                    data = pickle.load(file)
                    self.x_test.append(data[1])
                    self.y_test.append(data[0])

            except Exception as e: print(e)

    def compute_fim(self, num_samples=100):

        '''
        To compute the Fisher information matrix.
        '''

        if self.logger: self.logger.log(
            "Model.compute_fim",
            "Computing Fisher information matrix"
        )

        self.fim = [
            tf.zeros_like(layer)
            for layer in self.model.weights
        ]

        m = len(self.fim)
        n = len(self.x_train)
        
        for _ in range(num_samples):

            i = np.random.randint(n)
            j = np.random.randint(self.x_train[i].shape[0])

            with self.lock:
        
                with tf.GradientTape() as tape:

                    logits = tf.nn.log_softmax(
                        self.model(tf.expand_dims(self.x_train[i][j], axis=0))
                    )

                grads = tape.gradient(logits, self.model.weights)

                for i in range(m):
                    self.fim[i] += grads[i] ** 2  # type: ignore

        for i in range(m):
            self.fim[i] /= num_samples

    def timer(self, timeout: int):

        '''
        Timer to interrupt the traning procees.
        '''

        with self.interrupt_lock:
            self.interrupt = False

        if self.logger: self.logger.log(
            "Model.timer",
            "Interrupt timer started."
        )
            
        t, dt = 0, max(1, timeout // 100)
        while t < timeout:

            with self.interrupt_lock:
                if self.interrupt: break

            time.sleep(dt)
            t += dt
        
        with self.interrupt_lock:
            self.interrupt = True

        if self.logger: self.logger.log(
            "Model.timer",
            "Interrupt timer ended."
        )
    
    def train(
        self,
        num_seconds: float=-1,
        num_epochs: int=1,
        batch_size: int=256,
        dataset=None
    ) -> tuple [float, float]:

        '''
        Train the model for either set epochs or set seconds.
        '''
        
        if self.logger: self.logger.log(
            "Model.train",
            "Training for %s.",
            (
                ("%d epochs" % num_epochs) if num_seconds <= 0
                else ("%d seconds" % num_seconds)
            )
        )
        
        if dataset is None:

            n = self.x_train[0].shape[0]
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.x_train[0], self.y_train[0])
            )

            for x, y in zip(self.x_train[1:], self.y_train[1:]):
                
                n += x.shape[0]
                dataset = dataset.concatenate(
                    tf.data.Dataset.from_tensor_slices((x, y))
                )

            dataset = dataset.shuffle(n).batch(batch_size)

        '''
        If number of seconds is positive,
        create a timer to interrupt the training.
        '''

        timer: threading.Thread
        if num_seconds > 0:

            timer = threading.Thread(
                target=self.timer,
                args=(num_seconds,)
            )

            timer.start()

        '''
        Use these weights for Fedprox penalty.
        '''

        initial_weights = (
            [] if not self.prox_term
            else deepcopy(self.model.weights)
        )

        '''
        Train for 'num_epochs' epochs or 'num_seconds' seconds.
        '''

        batch = 0
        epoch = 0

        for e in self.tqdm(range(1000 if num_seconds > 0 else num_epochs)):

            epoch = e
            batch = 0
            
            for X_batch, Y_batch in dataset:  # type: ignore
                batch += 1

                with self.interrupt_lock:

                    if self.interrupt:
                        if self.logger:

                            self.logger.log(
                                "Model.train",
                                "Interrupted after %.2fs at epoch %d, batch %d",
                                num_seconds, e + 1, batch
                            )

                        break

                '''
                Compute gradients.
                '''

                with tf.GradientTape() as tape:
    
                    logits = self.model(X_batch, training=True)
                    loss = self.loss_fn(Y_batch, logits)

                    '''
                    Add FedProx penalty.
                    '''

                    if self.prox_term:

                        with self.lock:

                            k = tf.constant(0.5 * self.prox_term)

                            loss += k * tf.square(
                                tf.linalg.global_norm(
                                    tf.nest.map_structure(
                                        lambda a, b: a - b,
                                        self.model.weights,
                                        initial_weights
                                    )
                                )
                            )

                    '''
                    Add FedCurv penalty.
                    '''

                    if self.curv_term:

                        with self.lock:

                            n = len(self.fim)
                            if n > 0:

                                k = tf.constant(0.5 * self.curv_term)
                                ips = list(self.received_fims.keys())
                                np.random.shuffle(ips)

                                for ip in ips:

                                    with self.interrupt_lock:
                                        if self.interrupt: break

                                    for i in range(n):

                                        loss += k * tf.math.reduce_sum(
                                            self.received_fims[ip][i] * 
                                            tf.math.square(
                                                self.model.weights[i] - 
                                                self.received_weights[ip][i]
                                            )
                                        )

                '''
                Apply gradients.
                '''                

                grads = tape.gradient(
                    loss, self.model.trainable_variables
                )

                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                if not self.sync: self.merge(-1)

            else: continue

            '''
            If not timeout continue the loop,
            Else break out of the training loop.
            '''

            break

        if num_seconds > 0:

            with self.interrupt_lock:
                self.interrupt = True

            timer.join(num_seconds)  # type: ignore

        if self.logger: self.logger.log(
            "Model.train",
            "Finised at/after %.2f seconds %d epochs %d batches.",
            num_seconds, epoch + 1, batch
        )

        return self.test(
            set="train",
            dataset=dataset.rebatch(5000)
        )

    def test(
        self,
        set="eval",
        dataset=None
    ) -> tuple [float, float]:
        
        '''
        Evalute the model using dataset in {"train", "eval"}
        '''

        if dataset is None:

            n = self.x_test[0].shape[0]
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.x_test[0], self.y_test[0])
            )

            for x, y in zip(self.x_test[1:], self.y_test[1:]):
                
                n += x.shape[0]
                dataset = dataset.concatenate(
                    tf.data.Dataset.from_tensor_slices((x, y))
                )

            dataset = dataset.shuffle(n).batch(5000)

        acc, loss = 0, 0
        num_batches = 0

        for X_batch, Y_batch in dataset:  # type: ignore

            num_batches += 1
            logits = self.model(X_batch, training=False)
            loss += self.loss_fn(Y_batch, logits)  # type: ignore
            self.metric_fn.update_state(Y_batch, logits)
        
        loss = float(loss) / max(1.0, num_batches)
        acc = float(self.metric_fn.result())
        self.metric_fn.reset_state()

        if set == "eval":

            self.test_acc.append(acc)
            self.test_loss.append(loss)

            if self.logger: self.logger.log(
                "Model.test",
                "Testing finished with (acc, loss): {%.8f}, {%.8f}.",
                acc, loss
            )

        elif set == "train":

            self.train_acc.append(acc)
            self.train_loss.append(loss)

            if self.logger: self.logger.log(
                "Model.test",
                "Training finished with (acc, loss): {%.8f}, {%.8f}.",
                acc, loss
            )

        return acc, loss

    def merge(self, r: int) -> None:
        
        '''
        Merge the received models into a new consolidated model.

        r - merge models from round r.
        If r < 0 merge all the models.
        '''

        if r < 0:

            R = []
            with self.lock:
                R = list(self.model_queue.keys())

            for r in R: self.merge(r)
            return

        with self.lock:

            n = len(self.model_queue[r])
            n_ = 1.0 / float(n + 1)

            if n <= 0: return

            if self.logger: self.logger.log(
                "Model.merge",
                "Merging weights from %d devices",
                n
            )
                
            tf.nest.map_structure(
                lambda v, t: v.assign(t),
                self.model.weights,
                tf.nest.map_structure(
                    lambda *x: n_ * tf.add_n(x),  # type: ignore
                    * (
                        list(self.model_queue[r].values()) + 
                        [self.model.weights]
                    )
                )
            )

            self.model_queue.pop(r)
                

class MLP(Model):

    '''
    Overwrite 'create' method for each type of model.
    Here we create a multi layer perceptron,
    with layer size passed as args.
    '''

    def create(
        self,
        input_shape,
        num_outputs,
        learning_rate,
        l2_term,
        prox_term,
        curv_term,
        *args
    ) -> None:
        
        super().create(
            input_shape,
            num_outputs,
            learning_rate,
            l2_term,
            prox_term,
            curv_term,
            *args
        )

        self.model = keras.models.Sequential()
        
        self.model.add(
            keras.layers.Flatten(
                input_shape=input_shape
            )
        )

        for units in args:

            self.model.add(
                keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_term)
                )
            )

        self.model.add(
            keras.layers.Dense(
                units=num_outputs,
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        if self.random: self.model.set_weights(self.random.model)
        # self.model.summary()

    '''
    def compute_fim(self):

        if self.logger: self.logger.log(
            "Model.compute_fim",
            "Computing Fisher information matrix"
        )

        num_samples = 100
        n = len(self.x_train)
        
        weights = self.model.trainable_weights
        variance = [
            tf.zeros_like(tensor) 
            for tensor in weights
        ]
        
        for i in range(n):

            m = np.random.randint(
                self.x_train[i].shape[0] - num_samples
            )

            X = self.x_train[i][m:m+num_samples]
            for x in X:

                data = tf.expand_dims(x, axis=0)

                with tf.GradientTape() as tape:

                    output = self.model(data)
                    log_likelihood = tf.math.log(output)

                gradients = tape.gradient(log_likelihood, weights)
                variance = [
                    var + (grad ** 2) 
                    for var, grad in zip(variance, gradients)
                ]

        self.fim = [
            tensor / (num_samples * n) 
            for tensor in variance
        ]
    '''


class CNN(Model):

    '''
    Create a convolutional neural network with filters and layers from args.
    '''

    def create(
        self,
        input_shape,
        num_outputs,
        learning_rate,
        l2_term,
        prox_term,
        curv_term,
        *args
    ) -> None:
        
        super().create(
            input_shape,
            num_outputs,
            learning_rate,
            l2_term,
            prox_term,
            curv_term,
            *args
        )

        self.model = keras.models.Sequential()

        for filters in args[0]:

            self.model.add(
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(l2_term)
                )
            )

            self.model.add(
                keras.layers.MaxPooling2D()
            )
        
        self.model.add(
            keras.layers.Flatten()
        )

        for units in args[1]:

            self.model.add(
                keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_term)
                )
            )

        self.model.add(
            keras.layers.Dense(
                units=num_outputs,
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        if self.random: self.model.set_weights(self.random.model)
        # self.model.summary()


class LeNet5(Model):

    '''
    Create a LeNet5.
    '''

    def create(
        self,
        input_shape,
        num_outputs,
        learning_rate,
        l2_term,
        prox_term,
        curv_term,
        *args
    ) -> None:
        
        super().create(
            input_shape,
            num_outputs,
            learning_rate,
            l2_term,
            prox_term,
            curv_term,
            *args
        )

        self.model = keras.models.Sequential()

        self.model.add(
            keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                input_shape=input_shape,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        self.model.add(
            keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid"
            )
        )

        self.model.add(
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                padding="valid",
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        self.model.add(
            keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid"
            )
        )
        
        self.model.add(
            keras.layers.Flatten()
        )

        self.model.add(
            keras.layers.Dense(
                units=120,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        self.model.add(
            keras.layers.Dense(
                units=84,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        self.model.add(
            keras.layers.Dense(
                units=num_outputs,
                kernel_regularizer=keras.regularizers.l2(l2_term)
            )
        )

        if self.random: self.model.set_weights(self.random.model)
        # self.model.summary()


class ResNet(Model):
    
    '''Create a ResNet-20'''
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
    
    def create(
        self,
        input_shape,
        num_outputs,
        learning_rate,
        l2_term,
        prox_term,
        curv_term,
        *args
    ) -> None:

        super().create(
            input_shape,
            num_outputs,
            learning_rate,
            l2_term,
            prox_term,
            curv_term,
            *args
        )
        
        stack_n, initial_num_feature_maps, shortcut_type = args

        inputs = keras.layers.Input(shape=input_shape)
        
        x = keras.layers.Conv2D(
            initial_num_feature_maps, 
            kernel_size=(3,3),
            strides=(1,1),
            padding="same"
        )(inputs)
        
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Activation("relu")(x)
        
        filter_size = initial_num_feature_maps
        
        for layer_group in range(3):
            
            for block in range(stack_n):
                
                if layer_group > 0 and block == 0:
                    
                    filter_size *= 2
                    x = self.residual_block(x, filter_size, True, shortcut_type)
                
                else:
                    
                    x = self.residual_block(x, filter_size, False, shortcut_type)
        
        x = keras.layers.GlobalAveragePooling2D()(x)
        
        x = keras.layers.Flatten()(x)
        
        outputs = keras.layers.Dense(num_outputs)(x)

        self.model = keras.models.Model(
            inputs, outputs
        )
        
    @staticmethod
    def residual_block(x, number_of_filters, match_filter_size, shortcut_type):

        # Create skip connection
        
        x_skip = x

        # Perform the original mapping
        
        if match_filter_size:
            
            x = keras.layers.Conv2D(
                number_of_filters, 
                kernel_size=(3, 3), strides=(2,2),
                padding="same"
            )(x_skip)
            
        else:
            
            x = keras.layers.Conv2D(
                    number_of_filters,
                    kernel_size=(3, 3), 
                    strides=(1,1),
                    padding="same"
            )(x_skip)
            
        x = keras.layers.BatchNormalization(axis=3)(x)
        
        x = keras.layers.Activation("relu")(x)
        
        x = keras.layers.Conv2D(
            number_of_filters, 
            kernel_size=(3, 3),
            padding="same"
        )(x)
        
        x = keras.layers.BatchNormalization(axis=3)(x)

        # Perform matching of filter numbers if necessary
        
        if match_filter_size and shortcut_type == "identity":
            
            x_skip = keras.layers.Lambda(
                lambda x: tf.pad(
                    x[:, ::2, ::2, :], 
                    tf.constant([[0, 0,], [0, 0], [0, 0], [number_of_filters//4, number_of_filters//4]]),                 
                    mode="CONSTANT"
                )
            )(x_skip)
            
        elif match_filter_size and shortcut_type == "projection":
            
            x_skip = keras.layers.Conv2D(
                number_of_filters, 
                kernel_size=(1,1),
                strides=(2,2)
            )(x_skip)

        # Add the skip connection to the regular mapping
        
        x = keras.layers.Add()([x, x_skip])

        # Nonlinearly activate the result
        
        x = keras.layers.Activation("relu")(x)

        # Return the result
        return x


if __name__ == "__main__":

    '''
    Testing script.
    '''

    try:

        stack_n = 3
        initial_num_feature_maps = 16
        shortcut_type = "identity" # or: projection

        input_shape = [64, 64, 3]
        output_shape = 20

        logger = Logger("../../logs/resnet-test/")

        model = ResNet(logger, None, True)  # type: ignore
        model.create(
            input_shape, output_shape,
            0.01, 0, 0, 0,
            stack_n,
            initial_num_feature_maps,
            shortcut_type
        )

        train_files = ["../../datasets/tiny-imagenet/train/label_%d.b" % i for i in range(output_shape)]
        test_files = ["../../datasets/tiny-imagenet/test/label_%d.b" % i for i in range(output_shape)]

        model.preheat(
            train_files,
            test_files,
            (0, output_shape),
            (0, 0.1)
        )

        from tqdm import tqdm

        num_rounds = 5
        num_epochs = 3
        num_seconds = -1
        batch_size = 1024

        for r in tqdm(range(num_rounds)):

            print()
            print(*model.train(num_seconds, num_epochs, batch_size))
            print(*model.test())

        print()

        if input("enter [y] to dump results: ") == 'y':

            try:

                location = "../../results/resnet_test/"

                if not os.path.exists(location): 
                    os.makedirs(location)

                with open(
                    location + "ml.b", "wb"
                ) as file:
                    
                    result = (
                        [0] * num_rounds,
                        model.train_acc,
                        model.train_loss,
                        model.test_acc,
                        model.test_loss
                    )

                    pickle.dump(result, file)

                print("Dumped.")

            except Exception as e: print(e)

        print("Finished.")
        
    except Exception as e:
        print("Exited with error:", e)
    
