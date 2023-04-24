import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    for e in tf.train.summary_iterator(args.path):
        # for v in e.summary.value:
        step = e.step
        for v in e.summary.value:
            if v.tag == 'train_loss':
                print(step, v.simple_value)

    for e in tf.train.summary_iterator(args.path):
        # for v in e.summary.value:
        step = e.step
        for v in e.summary.value:
            if v.tag == 'train_loss_gtm':
                print(step, 'gtm', v.simple_value)