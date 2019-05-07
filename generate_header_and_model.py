import tensorflow as tf
import numpy as np
import argparse
import os
from models.model_espcn import ESPCN
from models.model_srcnn import SRCNN
from models.model_vespcn import VESPCN
from models.model_vsrnet import VSRnet
from collections import OrderedDict

@enum.unique
class Padding(enum.Enum):
    Valid = 0
    Same = 1
    Same_clamp_to_edge = 2

def get_arguments():
    parser = argparse.ArgumentParser(description='generate c header with model weights and binary model file')
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to use for generation')
    parser.add_argument('--output_folder', type=str, default='./',
                        help='where to put generated files')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint, from which weights are loaded')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether motion compensation is used in video super resolution model')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 3, 4],
                        help='What scale factor was used for chosen model')

    return parser.parse_args()


def dump_to_file(file, values, name):
    file.write('\nstatic const float ' + name + '[] = {\n')

    values_flatten = values.flatten()

    max_len = 0
    for value in values_flatten:
        if len(str(value)) > max_len:
            max_len = len(str(value))

    counter = 0
    for i in range(len(values_flatten)):
        counter += 1
        if counter == 4:
            file.write(str(values_flatten[i]) + 'f')
            if i != len(values_flatten) - 1:
                file.write(',')
            file.write('\n')
            counter = 0
        else:
            if counter == 1:
                file.write('    ')
            file.write(str(values_flatten[i]) + 'f')
            if i != len(values_flatten) - 1:
                file.write(',')
            file.write(' ' * (1 + max_len - len(str(values_flatten[i]))))
    if counter != 0:
        file.write('\n')
    file.write('};\n')

    file.write('\nstatic const long int ' + name + '_dims[] = {\n')
    for i in range(len(values.shape)):
        file.write('    ')
        file.write(str(values.shape[i]))
        if i != len(values.shape) - 1:
            file.write(',\n')
    file.write('\n};\n')


def write_conv_layer(kernel, bias, padding, activation, model_file):
    kernel = np.transpose(kernel, [3, 0, 1, 2])
    np.array([1, padding.value, activation, kernel.shape[3], kernel.shape[0], kernel.shape[1]], dtype=np.uint32).tofile(model_file)
    kernel.tofile(model_file)
    bias.tofile(model_file)


def write_depth_to_space_layer(block_size, model_file):
    np.array([2, block_size], dtype=np.uint32).tofile(model_file)


def prepare_native_mf_srcnn(weights, model_file):
    np.array([3], dtype=np.uint32).tofile(model_file)
    write_conv_layer(weights['srcnn/conv1/kernel:0'], weights['srcnn/conv1/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['srcnn/conv2/kernel:0'], weights['srcnn/conv2/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['srcnn/conv3/kernel:0'], weights['srcnn/conv3/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)


def prepare_native_mf_espcn(weights, model_file, scale_factor):
    np.array([4], dtype=np.uint32).tofile(model_file)
    write_conv_layer(weights['espcn/conv1/kernel:0'], weights['espcn/conv1/bias:0'], Padding.Same_clamp_to_edge, 1, model_file)
    write_conv_layer(weights['espcn/conv2/kernel:0'], weights['espcn/conv2/bias:0'], Padding.Same_clamp_to_edge, 1, model_file)
    write_conv_layer(weights['espcn/conv3/kernel:0'], weights['espcn/conv3/bias:0'], Padding.Same_clamp_to_edge, 2, model_file)
    write_depth_to_space_layer(scale_factor, model_file)


def prepare_native_mf_vespcn(weights, model_file, scale_factor):
    np.array([6], dtype=np.uint32).tofile(model_file)
    write_conv_layer(weights['vespcn/conv1/kernel:0'], weights['vespcn/conv1/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vespcn/conv2/kernel:0'], weights['vespcn/conv2/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vespcn/conv3/kernel:0'], weights['vespcn/conv3/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vespcn/conv4/kernel:0'], weights['vespcn/conv4/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vespcn/conv5/kernel:0'], weights['vespcn/conv5/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_depth_to_space_layer(scale_factor, model_file)


def prepare_native_mf_vsrnet(weights, model_file):
    np.array([3], dtype=np.uint32).tofile(model_file)
    write_conv_layer(weights['vsrnet/conv1/kernel:0'], weights['vsrnet/conv1/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vsrnet/conv2/kernel:0'], weights['vsrnet/conv2/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)
    write_conv_layer(weights['vsrnet/conv3/kernel:0'], weights['vsrnet/conv3/bias:0'], Padding.Same_clamp_to_edge, 0, model_file)


def main():
    args = get_arguments()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if args.ckpt_path is None:
        print("Path to the checkpoint file was not provided")
        exit(1)

    if args.model == 'srcnn':
        model = SRCNN(args)
    elif args.model == 'espcn':
        model = ESPCN(args)
    elif args.model == 'vespcn':
        model = VESPCN(args)
    elif args.model == 'vsrnet':
        model = VSRnet(args)
    else:
        exit(1)

    with tf.Session() as sess:
        input_ph = model.get_placeholder()
        predicted = model.load_model(input_ph)

        if args.model == 'vespcn':
            predicted = predicted[2]
        predicted = tf.identity(predicted, name='y')

        if os.path.isdir(args.ckpt_path):
            args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_path)

        with open(os.path.join(args.output_folder, args.model + '.model'), 'wb') as native_mf:
            weights = model.get_model_weights(sess)
            if args.model == 'srcnn':
                prepare_native_mf_srcnn(weights, native_mf)
            elif args.model == 'espcn':
                prepare_native_mf_espcn(weights, native_mf, args.scale_factor)
            elif args.model == 'vespcn':
                prepare_native_mf_vespcn(weights, native_mf, args.scale_factor)
            elif args.model == 'vsrnet':
                prepare_native_mf_vsrnet(weights, native_mf)

        with open(os.path.join(args.output_folder, 'dnn_' + args.model + '.h'), 'w') as header:
            header.write('/**\n')
            header.write(' * @file\n')
            header.write(' * Default cnn weights for x' + str(args.scale_factor) + ' upscaling with ' +
                         args.model + ' model.\n')
            header.write(' */\n\n')

            header.write('#ifndef AVFILTER_DNN_' + args.model.upper() + '_H\n')
            header.write('#define AVFILTER_DNN_' + args.model.upper() + '_H\n')

            variables = tf.trainable_variables()
            var_dict = OrderedDict()
            for variable in variables:
                var_name = variable.name.split(':')[0].replace('/', '_')
                value = variable.eval()
                if 'kernel' in var_name:
                    value = np.transpose(value, axes=(3, 0, 1, 2))
                var_dict[var_name] = value

            for name, value in var_dict.items():
                dump_to_file(header, value, name)

            header.write('#endif\n')

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
        tf.train.write_graph(output_graph_def, args.output_folder, args.model + '.pb', as_text=False)


if __name__ == '__main__':
    main()

