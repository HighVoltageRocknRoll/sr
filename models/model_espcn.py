import tensorflow as tf
from .model import Model
from .dataset import Dataset


class ESPCN(Model):
    def __init__(self, args):
        super().__init__(args)
        self._save_num = args.save_num
        if hasattr(args, 'shuffle_buffer_size'):
            self.dataset = Dataset(args.batch_size, args.dataset_path, args.dataset_info_path, args.shuffle_buffer_size)
        else:
            self.dataset = Dataset(args.batch_size, args.dataset_path, args.dataset_info_path)
        self._prediction_offset = self.dataset.scale_factor * 4

    def get_data(self):
        data_batch, initializer = self.dataset.get_data()

        lr_batch = tf.cast(data_batch['lr1'], tf.float32) / 255.0
        hr_batch = tf.cast(data_batch['hr'], tf.float32) / 255.0

        return [lr_batch, hr_batch], initializer

    def load_model(self, data_batch):
        lr_batch = data_batch[0]

        with tf.variable_scope('espcn'):
            net = tf.layers.conv2d(lr_batch, 64, 5, activation=tf.nn.tanh, padding='valid', name='conv1',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 32, 3, activation=tf.nn.tanh, padding='valid', name='conv2',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, self.dataset.scale_factor ** 2, 3, activation=tf.nn.sigmoid, padding='valid',
                                   name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
            predicted_batch = tf.depth_to_space(net, self.dataset.scale_factor, name='prediction')

        espcn_variables = tf.trainable_variables(scope='espcn')
        for variable in espcn_variables:
            if 'conv3' in variable.name:
                self.lr_multipliers[variable.name] = 0.1
            else:
                self.lr_multipliers[variable.name] = 1.0

        tf.summary.image('Low_resolution', data_batch[0][:, 4:-4, 4:-4], max_outputs=self._save_num)
        tf.summary.image('High_resolution',
                         data_batch[1][:, self._prediction_offset:-self._prediction_offset,
                                       self._prediction_offset:-self._prediction_offset],
                         max_outputs=self._save_num)
        tf.summary.image('High_resolution_prediction', predicted_batch, max_outputs=self._save_num)

        return predicted_batch

    def get_loss(self, data_batch, predicted_batch):
        loss = tf.losses.mean_squared_error(
            data_batch[1][:,
                          self._prediction_offset:-self._prediction_offset,
                          self._prediction_offset:-self._prediction_offset],
            predicted_batch)

        tf.summary.scalar('MSE', loss)
        tf.summary.scalar('PSNR', tf.reduce_mean(tf.image.psnr(
                                                     data_batch[1][:,
                                                                   self._prediction_offset:-self._prediction_offset,
                                                                   self._prediction_offset:-self._prediction_offset],
                                                     predicted_batch,
                                                     max_val=1.0)))
        tf.summary.scalar('SSIM', tf.reduce_mean(tf.image.ssim(
                                                     data_batch[1][:,
                                                                   self._prediction_offset:-self._prediction_offset,
                                                                   self._prediction_offset:-self._prediction_offset],
                                                     predicted_batch,
                                                     max_val=1.0)))

        return loss

    def calculate_metrics(self, data_batch, predicted_batch):
        diff = data_batch[1][:, self._prediction_offset:-self._prediction_offset,
               self._prediction_offset:-self._prediction_offset] - predicted_batch
        diff_sqr = tf.square(diff)

        mse = ('MSE', tf.reduce_mean(diff_sqr, axis=[1, 2, 3]))
        psnr = ('PSNR', tf.squeeze(tf.image.psnr(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predicted_batch,
                                       max_val=1.0)))
        ssim = ('SSIM', tf.squeeze(tf.image.ssim(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predicted_batch,
                                       max_val=1.0)))

        return [mse, psnr, ssim]
