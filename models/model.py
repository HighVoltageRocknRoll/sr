from abc import ABC, abstractmethod
import tensorflow as tf
import json


class Model(ABC):
    @abstractmethod
    def __init__(self, args):
        self.lr_multipliers = {}

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def load_model(self, data):
        pass

    @abstractmethod
    def get_loss(self, data_batch, predicted_batch):
        pass

    @abstractmethod
    def calculate_metrics(self, data_batch, predicted_batch):
        pass

    @staticmethod
    def dump_model_weights(output_path, sess):
        variables = dict([(var.name, sess.run(var).flatten().tolist()) for var in tf.trainable_variables()])
        with open(output_path, 'w') as json_file:
            json.dump(variables, json_file, indent=4)
