import tensorflow as tf
import numpy as np
import argparse
import os
from models.model_espcn import ESPCN
from models.model_srcnn import SRCNN
from models.model_vespcn import VESPCN
from models.model_vsrnet import VSRnet
from collections import OrderedDict


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

        if args.use_mc:
            predicted = predicted[2]
        predicted = tf.identity(predicted, name='y')

        if os.path.isdir(args.ckpt_path):
            args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_path)

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

            header.write('\nstatic const float* ' + args.model + '_consts[] = {\n')
            names = list(var_dict.keys())
            for i in range(len(names)):
                header.write('    ' + names[i])
                if i != len(names) - 1:
                    header.write(',')
                header.write('\n')
            header.write('};\n')

            header.write('\nstatic const long int* ' + args.model + '_consts_dims[] = {\n')
            for i in range(len(names)):
                header.write('    ' + names[i] + '_dims')
                if i != len(names) - 1:
                    header.write(',')
                header.write('\n')
            header.write('};\n')

            header.write('\nstatic const int ' + args.model + '_consts_dims_len[] = {\n')
            for i in range(len(names)):
                header.write('    ' + str(len(var_dict[names[i]].shape)))
                if i != len(names) - 1:
                    header.write(',')
                header.write('\n')
            header.write('};\n')

            activations = ['Relu', 'Tanh', 'Sigmoid']
            activations_in_graph = [n.name.split('/')[-1] for n in tf.get_default_graph().as_graph_def().node
                                    if n.name.split('/')[-1] in activations]
            unique_activations = []
            for activation in activations_in_graph:
                if activation not in unique_activations:
                    unique_activations.append(activation)
                    header.write('\nstatic const char ' + args.model + '_' + activation.lower() + '[] = "'
                                 + activation + '";\n')

            header.write('\nstatic const char* ' + args.model + '_activations[] = {\n')
            for i in range(len(activations_in_graph)):
                header.write('    ' + args.model + '_' + activations_in_graph[i].lower())
                if i != len(activations_in_graph) - 1:
                    header.write(',')
                header.write('\n')
            header.write('};\n\n')

            header.write('#endif\n')

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
        tf.train.write_graph(output_graph_def, args.output_folder, args.model + '.pb', as_text=False)


if __name__ == '__main__':
    main()

