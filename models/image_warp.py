import tensorflow as tf


def image_warp(images, flow, name='image_warp'):
    with tf.name_scope(name):
        shape = tf.shape(images)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        grid = tf.expand_dims(tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype), axis=0)
        coords = tf.reshape(grid + flow, [batch_size, height * width, 2])
        coords = tf.stack([tf.minimum(tf.maximum(0.0, coords[:, :, 0]), tf.cast(height, flow.dtype) - 1.0),
                           tf.minimum(tf.maximum(0.0, coords[:, :, 1]), tf.cast(width, flow.dtype) - 1.0)], axis=2)

        floors = tf.cast(tf.floor(coords), tf.int32)
        ceils = floors + 1
        alphas = tf.cast(coords - tf.cast(floors, flow.dtype), images.dtype)
        alphas = tf.reshape(tf.minimum(tf.maximum(0.0, alphas), 1.0), shape=[batch_size, height, width, 1, 2])

        images_flattened = tf.reshape(images, [-1, channels])
        batch_offsets = tf.expand_dims(tf.range(batch_size) * height * width, axis=1)

        def gather(y_coords, x_coords):
            linear_coordinates = batch_offsets + y_coords * width + x_coords
            gathered_values = tf.gather(images_flattened, linear_coordinates)
            return tf.reshape(gathered_values, shape)

        top_left = gather(floors[:, :, 0], floors[:, :, 1])
        top_right = gather(floors[:, :, 0], ceils[:, :, 1])
        bottom_left = gather(ceils[:, :, 0], floors[:, :, 1])
        bottom_right = gather(ceils[:, :, 0], ceils[:, :, 1])

        interp_top = alphas[:, :, :, :, 1] * (top_right - top_left) + top_left
        interp_bottom = alphas[:, :, :, :, 1] * (bottom_right - bottom_left) + bottom_left
        interpolated = alphas[:, :, :, :, 0] * (interp_bottom - interp_top) + interp_top

        interpolated = tf.reshape(interpolated, shape)

        return interpolated