try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from .model import Model
from .image_warp import image_warp


class VESPCN(Model):
    def __init__(self, args):
        super().__init__(args)
        self._prediction_offset = self._scale_factor * 5
        self._use_mc = args.use_mc
        if hasattr(args, 'mc_independent'):
            self._mc_independent = args.mc_independent
        else:
            self._mc_independent = False

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
        if self._use_mc:
            with tf.variable_scope('mc'):
                neighboring_frames = tf.expand_dims(tf.concat([data_batch[0][:, :, :, 0], data_batch[0][:, :, :, 2]],
                                                              axis=0), axis=3)
                lr_input = tf.concat([tf.stack([data_batch[0][:, :, :, 1], data_batch[0][:, :, :, 0]], axis=3),
                                      tf.stack([data_batch[0][:, :, :, 1], data_batch[0][:, :, :, 2]], axis=3)], axis=0)
                with tf.variable_scope('coarse_flow'):
                    net = tf.layers.conv2d(lr_input, 24, 5, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv1', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv2', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv4', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 32, 3, strides=1, activation=tf.nn.tanh, padding='same',
                                           name='conv5', kernel_initializer=tf.keras.initializers.he_normal())
                    coarse_flow = 36.0 * tf.depth_to_space(net, 4)

                    if self._using_dataset:
                        tf.summary.image('Coarse_flow_y', (coarse_flow[:, :, :, 0:1] + 36.0) / 72.0,
                                         max_outputs=self._save_num)
                        tf.summary.image('Coarse_flow_x', (coarse_flow[:, :, :, 1:2] + 36.0) / 72.0,
                                         max_outputs=self._save_num)

                    warped_frames = image_warp(neighboring_frames, coarse_flow)

                ff_input = tf.concat([lr_input, coarse_flow, warped_frames], axis=3)
                with tf.variable_scope('fine_flow'):
                    net = tf.layers.conv2d(ff_input, 24, 5, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv1', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv2', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv4', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 8, 3, strides=1, activation=tf.nn.tanh, padding='same',
                                           name='conv5', kernel_initializer=tf.keras.initializers.he_normal())
                    fine_flow = 36.0 * tf.depth_to_space(net, 2)
                    flow = coarse_flow + fine_flow

                    if self._using_dataset:
                        tf.summary.image('Flow_y', (flow[:, :, :, 0:1] + 36.0) / 72.0,
                                         max_outputs=self._save_num)
                        tf.summary.image('Flow_x', (flow[:, :, :, 1:2] + 36.0) / 72.0,
                                         max_outputs=self._save_num)

                    warped_frames = image_warp(neighboring_frames, flow)
            if self._mc_independent:
                sr_input = tf.concat([tf.stop_gradient(warped_frames[:tf.shape(data_batch[0])[0]]),
                                      data_batch[0][:, :, :, 1:2],
                                      tf.stop_gradient(warped_frames[tf.shape(data_batch[0])[0]:])], axis=3)
            else:
                sr_input = tf.concat([warped_frames[:tf.shape(data_batch[0])[0]],
                                      data_batch[0][:, :, :, 1:2],
                                      warped_frames[tf.shape(data_batch[0])[0]:]], axis=3)

            if self._using_dataset:
                tf.summary.image('Low_resolution_warped0', tf.expand_dims(sr_input[:, 5:-5, 5:-5, 0], axis=3),
                                 max_outputs=self._save_num)
                tf.summary.image('Low_resolution_warped2', tf.expand_dims(sr_input[:, 5:-5, 5:-5, 2], axis=3),
                                 max_outputs=self._save_num)
        else:
            flow = []
            warped_frames = []
            sr_input = data_batch[0]

        with tf.variable_scope('vespcn'):
            if not self._using_dataset:
                sr_input = tf.pad(sr_input, [[0, 0], [5, 5], [5, 5], [0, 0]], 'SYMMETRIC')
            net = tf.layers.conv2d(sr_input, 24, 3, activation=tf.nn.relu, padding='valid', name='conv1',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 24, 3, activation=tf.nn.relu, padding='valid', name='conv2',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 24, 3, activation=tf.nn.relu, padding='valid', name='conv3',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 24, 3, activation=tf.nn.relu, padding='valid', name='conv4',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, self._scale_factor ** 2, 3, activation=None, padding='valid',
                                   name='conv5', kernel_initializer=tf.keras.initializers.he_normal())
            predicted_batch = tf.depth_to_space(net, self._scale_factor, name='prediction')

        if self._using_dataset:
            tf.summary.image('Low_resolution0', tf.expand_dims(data_batch[0][:, 5:-5, 5:-5, 0], axis=3),
                             max_outputs=self._save_num)
            tf.summary.image('Low_resolution1', tf.expand_dims(data_batch[0][:, 5:-5, 5:-5, 1], axis=3),
                             max_outputs=self._save_num)
            tf.summary.image('Low_resolution2', tf.expand_dims(data_batch[0][:, 5:-5, 5:-5, 2], axis=3),
                             max_outputs=self._save_num)
            tf.summary.image('High_resolution',
                             data_batch[1][:, self._prediction_offset:-self._prediction_offset,
                                           self._prediction_offset:-self._prediction_offset],
                             max_outputs=self._save_num)
            tf.summary.image('High_resolution_prediction', predicted_batch, max_outputs=self._save_num)

        return flow, warped_frames, predicted_batch

    def get_loss(self, data_batch, predicted_batch):
        flow, warped_frames, predictions = predicted_batch

        mse_loss = tf.losses.mean_squared_error(
            data_batch[1][:,
                          self._prediction_offset:-self._prediction_offset,
                          self._prediction_offset:-self._prediction_offset],
            predictions)

        tf.summary.scalar('MSE', mse_loss)
        tf.summary.scalar('PSNR', tf.reduce_mean(tf.image.psnr(
            data_batch[1][:,
                self._prediction_offset:-self._prediction_offset,
                self._prediction_offset:-self._prediction_offset],
            predictions,
            max_val=1.0)))
        tf.summary.scalar('SSIM', tf.reduce_mean(tf.image.ssim(
            data_batch[1][:,
                self._prediction_offset:-self._prediction_offset,
                self._prediction_offset:-self._prediction_offset],
            predictions,
            max_val=1.0)))

        if self._use_mc:
            cur_frames = tf.expand_dims(tf.concat([data_batch[0][:, :, :, 1], data_batch[0][:, :, :, 1]], axis=0), axis=3)
            warp_loss = tf.losses.mean_squared_error(cur_frames, warped_frames)

            grad_x_kernel = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32, shape=(1, 3, 1, 1))
            grad_y_kernel = tf.constant([-1.0, 0, 1.0], dtype=tf.float32, shape=(3, 1, 1, 1))
            flow = tf.expand_dims(tf.concat([flow[:, :, :, 0], flow[:, :, :, 1]], axis=0), axis=3)
            flow_grad_x = tf.nn.conv2d(flow, grad_x_kernel, [1, 1, 1, 1], padding='VALID')[:, 1:-1, :, :]
            flow_grad_y = tf.nn.conv2d(flow, grad_y_kernel, [1, 1, 1, 1], padding='VALID')[:, :, 1:-1, :]
            huber_loss = tf.sqrt(0.01 + tf.reduce_sum(flow_grad_x * flow_grad_x + flow_grad_y * flow_grad_y))

            tf.summary.scalar('Warp_loss', warp_loss)
            tf.summary.scalar('Huber_loss', huber_loss)

            if self._mc_independent:
                return mse_loss + warp_loss + 0.01 * huber_loss
            else:
                return mse_loss + 0.01 * warp_loss + 0.001 * huber_loss
        else:
            return mse_loss

    def calculate_metrics(self, data_batch, predicted_batch):
        flow, warped_frames, predictions = predicted_batch
        diff = data_batch[1][:, self._prediction_offset:-self._prediction_offset,
               self._prediction_offset:-self._prediction_offset] - predictions
        diff_sqr = tf.square(diff)

        mse = ('MSE', tf.reduce_mean(diff_sqr, axis=[1, 2, 3]))
        psnr = ('PSNR', tf.squeeze(tf.image.psnr(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predictions,
                                       max_val=1.0)))
        ssim = ('SSIM', tf.squeeze(tf.image.ssim(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predictions,
                                       max_val=1.0)))

        return [mse, psnr, ssim]
