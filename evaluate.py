try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
from time import time
from models.model_espcn import ESPCN
from models.model_srcnn import SRCNN
from models.model_vespcn import VESPCN
from models.model_vsrnet import VSRnet

BATCH_SIZE = 4
DATASET_PATH = 'dataset.tfrecords'
DATASET_INFO_PATH = 'dataset_info.txt'
SAVE_NUM = 2
LOGDIR = 'evaluation_logdir/default'
STEPS_PER_LOG = 5


def get_arguments():
    parser = argparse.ArgumentParser(description='evaluate one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to evaluate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--dataset_info_path', type=str, default=DATASET_INFO_PATH,
                        help='Path to the dataset info')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save image summaries')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether to use motion compensation in video super resolution model')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save summaries')

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

        metrics = model.calculate_metrics(data_batch, predicted_batch)

        if args.ckpt_path is None:
            print("Path to the checkpoint file was not provided")
            exit(1)

        if os.path.isdir(args.ckpt_path):
            args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_path)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

        sess.run(data_initializer)

        steps = model.dataset.examples_num // args.batch_size + (1 if model.dataset.examples_num % args.batch_size > 0
                                                                 else 0)
        epoch = int(args.ckpt_path.split('.')[0].split('_')[-1])
        logged_iterations = 0
        metrics_results = [[metric[0], np.array([])] for metric in metrics]
        time_res = 0.0
        for i in tqdm(range(steps), total=steps, unit='step'):
            start = time()
            results = sess.run([metric[1] for metric in metrics] + [summary])
            time_res += time() - start
            cur_metrics_results = results[:-1]
            for j in range(len(cur_metrics_results)):
                if len(cur_metrics_results[j].shape) == len(metrics_results[j][1].shape):
                    metrics_results[j][1] = np.concatenate((metrics_results[j][1], cur_metrics_results[j]))
                else:
                    metrics_results[j][1] = np.concatenate((metrics_results[j][1], [cur_metrics_results[j]]))
            cur_summary_results = results[-1]
            if (i + 1) % args.steps_per_log == 0:
                summary_writer.add_summary(cur_summary_results, epoch * steps + logged_iterations)
                logged_iterations += 1

        mean_metrics = [(metric[0], np.mean(metric[1])) for metric in metrics_results]
        mean_metrics.append(("Time", time_res / model.dataset.examples_num))
        metric_summaries = []
        for metric in mean_metrics:
            print("Mean " + metric[0] + ': ', metric[1])
            metric_summaries.append(tf.summary.scalar(metric[0], metric[1]))

        metric_summary = tf.summary.merge(metric_summaries)
        metric_summary_res = sess.run(metric_summary)
        summary_writer.add_summary(metric_summary_res, epoch)


if __name__ == '__main__':
    main()
