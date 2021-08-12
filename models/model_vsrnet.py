try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from .model import Model


class VSRnet(Model):
    def __init__(self, args):
        super().__init__(args)
        self._prediction_offset = 8
        self._lr_offset = self._prediction_offset // self._scale_factor

    def get_data(self):
        data_batch, initializer = self.dataset.get_data()

        lr_batch = tf.concat([tf.cast(data_batch['lr0'], tf.float32) / 255.0,
                              tf.cast(data_batch['lr1'], tf.float32) / 255.0,
                              tf.cast(data_batch['lr2'], tf.float32) / 255.0], axis=3)
        hr_batch = tf.cast(data_batch['hr'], tf.float32) / 255.0

        return [lr_batch, hr_batch], initializer

    def get_placeholder(self):
        input_ph = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="x")

        return [input_ph]

    def load_model(self, data_batch):
        lr_batch = data_batch[0]

        with tf.variable_scope('vsrnet'):
            if self._using_dataset:
                net = tf.image.resize_bicubic(lr_batch, (self._scale_factor * lr_batch.shape[1],
                                                         self._scale_factor * lr_batch.shape[2]), align_corners=True)
            else:
                net = tf.pad(lr_batch, [[0, 0], [8, 8], [8, 8], [0, 0]], 'SYMMETRIC')
            net = tf.layers.conv2d(net, 64, 9, activation=tf.nn.relu, padding='valid', name='conv1',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 32, 5, activation=tf.nn.relu, padding='valid', name='conv2',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 1, 5, activation=tf.nn.relu, padding='valid',
                                   name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
            predicted_batch = tf.maximum(net, 0.0)

        vsrnet_variables = tf.trainable_variables(scope='vsrnet')
        for variable in vsrnet_variables:
            if 'conv3' in variable.name:
                self.lr_multipliers[variable.name] = 0.1
            else:
                self.lr_multipliers[variable.name] = 1.0

        if self._using_dataset:
            tf.summary.image('Low_resolution0',
                             tf.expand_dims(data_batch[0][:, self._lr_offset:-self._lr_offset,
                                                          self._lr_offset:-self._lr_offset, 0], axis=3),
                             max_outputs=self._save_num)
            tf.summary.image('Low_resolution1',
                             tf.expand_dims(data_batch[0][:, self._lr_offset:-self._lr_offset,
                                                          self._lr_offset:-self._lr_offset, 1], axis=3),
                             max_outputs=self._save_num)
            tf.summary.image('Low_resolution2',
                             tf.expand_dims(data_batch[0][:, self._lr_offset:-self._lr_offset,
                                                          self._lr_offset:-self._lr_offset, 2], axis=3),
                             max_outputs=self._save_num)
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
