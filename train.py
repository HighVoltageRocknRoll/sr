try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import argparse
from tqdm import tqdm
import os
from models.model_espcn import ESPCN
from models.model_srcnn import SRCNN
from models.model_vespcn import VESPCN
from models.model_vsrnet import VSRnet


MODEL='srcnn'
BATCH_SIZE = 32
DATASET_PATH = 'dataset.tfrecords'
DATASET_INFO_PATH = 'dataset_info.txt'
SHUFFLE_BUFFER_SIZE = 300000
OPTIMIZER='adam'
LEARNING_RATE = 1e-4
LEARNING_DECAY_RATE = 1e-1
LEARNING_DECAY_EPOCHS = 40
MOMENTUM = 0.9
NUM_EPOCHS = 100
SAVE_NUM = 2
STEPS_PER_LOG = 100
EPOCHS_PER_SAVE = 1
LOGDIR = 'training_logdir/default'


def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['srcnn', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--dataset_info_path', type=str, default=DATASET_INFO_PATH,
                        help='Path to the dataset info')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--shuffle_buffer_size', type=int, default=SHUFFLE_BUFFER_SIZE,
                        help='Buffer size used for shuffling examples in dataset')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, choices=['adam', 'momentum', 'sgd'],
                        help='What optimizer to use for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--use_lr_decay', action='store_true',
                        help='Whether to apply exponential decay to the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=LEARNING_DECAY_RATE,
                        help='Learning rate decay rate used in exponential decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=LEARNING_DECAY_EPOCHS,
                        help='Number of epochs before full decay rate tick used in exponential decay')
    parser.add_argument('--staircase_lr_decay', action='store_true',
                        help='Whether to decay the learning rate at discrete intervals')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save summaries')
    parser.add_argument('--epochs_per_save', type=int, default=EPOCHS_PER_SAVE,
                        help='How often to save checkpoints')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether to use motion compensation in video super resolution model')
    parser.add_argument('--mc_independent', action='store_true',
                        help='Whether to train motion compensation network independent from super resolution network')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save checkpoints and summaries')

    return parser.parse_args()


def main():
    args = get_arguments()

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
        data_batch, data_initializer = model.get_data()

        predicted_batch = model.load_model(data_batch)

        loss = model.get_loss(data_batch, predicted_batch)

        global_step = tf.Variable(0, trainable=False)
        if args.use_lr_decay:
            lr = tf.train.exponential_decay(args.learning_rate,
                                            global_step,
                                            args.lr_decay_epochs * model.dataset.examples_num,
                                            args.lr_decay_rate,
                                            staircase=args.staircase_lr_decay)

        else:
            lr = args.learning_rate
        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif args.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, args.momentum)
        elif args.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        grads_vars = optimizer.compute_gradients(loss)
        grads_vars_final = []
        for gradient, variable in grads_vars:
            assert gradient is not None, variable.name

            if variable.name in model.lr_multipliers.keys():
                gradient *= model.lr_multipliers[variable.name]
            grads_vars_final.append((gradient, variable))

            variable_name = variable.name.replace(':', '_')

            scope = 'TrainLogs/' + variable_name + '/Values/'
            tf.summary.scalar(scope + 'MIN', tf.reduce_min(variable))
            tf.summary.scalar(scope + 'MAX', tf.reduce_max(variable))
            tf.summary.scalar(scope + 'L2', tf.norm(variable))
            tf.summary.scalar(scope + 'AVG', tf.reduce_mean(variable))

            scope = 'TrainLogs/' + variable_name + '/Gradients/'
            tf.summary.scalar(scope + 'MIN', tf.reduce_min(gradient))
            tf.summary.scalar(scope + 'MAX', tf.reduce_max(gradient))
            tf.summary.scalar(scope + 'L2', tf.norm(gradient))
            tf.summary.scalar(scope + 'AVG', tf.reduce_mean(gradient))
        train_op = optimizer.apply_gradients(grads_vars_final, global_step=global_step)
        tf.summary.scalar('Learning_rate', lr)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

        saver = tf.train.Saver()
        last_epoch = 0
        if args.ckpt_path is None:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
        else:
            if os.path.isdir(args.ckpt_path):
                args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
            last_epoch = int(args.ckpt_path.split('.')[0].split('_')[-1])
            saver.restore(sess, args.ckpt_path)
        sess.run(data_initializer)

        num_steps_in_epoch = model.dataset.examples_num // args.batch_size + \
                             1 if model.dataset.examples_num % args.batch_size != 0 else 0
        for epoch in range(args.num_epochs):
            print("Epoch: ", epoch + last_epoch)
            bar = tqdm(range(num_steps_in_epoch),
                       total=num_steps_in_epoch,
                       unit='step',
                       smoothing=1.0)
            for i in bar:
                _, cur_loss, cur_summary, = sess.run([train_op, loss, summary])
                bar.set_description('Loss: ' + str(cur_loss))
                bar.refresh()
                if (i + 1) % args.steps_per_log == 0:
                    summary_writer.add_summary(cur_summary, (last_epoch + epoch) * num_steps_in_epoch + i)
            if epoch % args.epochs_per_save == 0:
                saver.save(sess, os.path.join(args.logdir, 'model_' + str(last_epoch + epoch + 1) + '.ckpt'))


if __name__ == '__main__':
    main()
