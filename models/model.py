from abc import ABC, abstractmethod
import tensorflow as tf
import json
from .dataset import Dataset

class Model(ABC):
    @abstractmethod
    def __init__(self, args):
        self.lr_multipliers = {}
        if hasattr(args, 'dataset_path') and hasattr(args, 'dataset_info_path'):
            if hasattr(args, 'shuffle_buffer_size'):
                self.dataset = Dataset(args.batch_size,
                                       args.dataset_path,
                                       args.dataset_info_path,
                                       args.shuffle_buffer_size)
            else:
                self.dataset = Dataset(args.batch_size,
                                       args.dataset_path,
                                       args.dataset_info_path)
            self._scale_factor = self.dataset.scale_factor
            if hasattr(args, 'save_num'):
                self._save_num = args.save_num
            else:
                self._save_num = 1
            self._using_dataset = True
        else:
            self._using_dataset = False
            self._scale_factor = args.scale_factor

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_placeholder(self):
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
    def get_model_weights(sess):
        return dict([(var.name, sess.run(var)) for var in tf.trainable_variables()])

    @staticmethod
    def dump_model_weights(output_path, sess):
        variables = dict([(var.name, sess.run(var).flatten().tolist()) for var in tf.trainable_variables()])
        with open(output_path, 'w') as json_file:
            json.dump(variables, json_file, indent=4)
