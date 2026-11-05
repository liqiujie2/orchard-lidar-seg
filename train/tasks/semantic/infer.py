#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(project_root)
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import shutil
import __init__ as booger

from tasks.semantic.modules.user import *
import tasks.semantic.modules.user as user

print("✅ Loaded user.py from:", user.__file__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
                datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("----------\n")

    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(os.path.join(FLAGS.model, "arch_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(os.path.join(FLAGS.model, "data_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    try:
        FLAGS.log = os.path.normpath(FLAGS.log)
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))

        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            seq_dir = os.path.join(FLAGS.log, "sequences", seq)
            os.makedirs(seq_dir)
            os.makedirs(os.path.join(seq_dir, "predictions"))

        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            seq_dir = os.path.join(FLAGS.log, "sequences", seq)
            os.makedirs(seq_dir)
            os.makedirs(os.path.join(seq_dir, "predictions"))

        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            seq_dir = os.path.join(FLAGS.log, "sequences", seq)
            os.makedirs(seq_dir)
            os.makedirs(os.path.join(seq_dir, "predictions"))

    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model)
    user.infer()