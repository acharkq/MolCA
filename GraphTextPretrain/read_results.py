import argparse
import tensorflow.compat.v1 as tf
from pathlib import Path
tf.disable_v2_behavior()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    args = parser.parse_args()
    if not args.path.find('events.out.tfevents') >= 0:
        args.path = str(list(Path(args.path).glob('events.out.tfevents*'))[0])

    for e in tf.train.summary_iterator(args.path):
        # for v in e.summary.value:
        step = e.step
        for v in e.summary.value:
            if v.tag == args.tag:
                print(step, v.simple_value)
        if step > args.max_step:
            break

    # for e in tf.train.summary_iterator(args.path):
    #     # for v in e.summary.value:
    #     step = e.step
    #     for v in e.summary.value:
    #         if v.tag == 'train_loss_gtm':
    #             print(step, 'gtm', v.simple_value)