#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import torch
import __init__ as booger

from tasks.semantic.modules.ioueval import iouEval
import tasks.semantic.modules.ioueval as iou_module

print(">>> [DEBUG] iouEval is imported from:", iou_module.__file__)
print(">>> [DEBUG] iouEval has get_precision_recall_f1:",
      hasattr(iou_module.iouEval, 'get_precision_recall_f1'))

from common.laserscan import SemLaserScan

# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_iou.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default="config/labels/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit', '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud. Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("Distance split: 1m interval:0~1,1~2,2~3,3~4,4~5,5~6,6~7,7~8,8~9,9~10 + All")
  print("*" * 80)

  assert(FLAGS.split in splits)

  try:
    print("Opening data config file %s" % FLAGS.data_cfg)
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  class_strings = DATA["labels"]
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)

  maxkey = max(class_remap.keys()) + 100
  remap_lut = np.zeros((maxkey), dtype=np.int32)
  for key, data in class_remap.items():
    remap_lut[key] = data

  ignore = []
  for cl, ign in class_ignore.items():
    if ign:
      x_cl = int(cl)
      ignore.append(x_cl)
      print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

  device = torch.device("cpu")

  evaluator_all = iouEval(nr_classes, device, ignore)
  evaluator_all.reset()

  range_list = [(i, i+1) for i in range(10)]
  eval_dict = {}
  for s, e in range_list:
      eval_dict[(s,e)] = iouEval(nr_classes, device, ignore)
      eval_dict[(s,e)].reset()

  test_sequences = DATA["split"][FLAGS.split]

  scan_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    scan_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "velodyne")
    seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
    seq_scan_names.sort()
    scan_names.extend(seq_scan_names)

  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
    seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn if ".label" in f]
    seq_label_names.sort()
    label_names.extend(seq_label_names)

  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
    seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
    seq_pred_names.sort()
    pred_names.extend(seq_pred_names)

  assert(len(label_names) == len(scan_names) and len(label_names) == len(pred_names))

  print("Evaluating sequences: ")
  for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
    print("evaluating label ", label_file, "with", pred_file)

    scan = SemLaserScan(project=False)
    scan.open_scan(scan_file)
    pts = scan.points

    label = SemLaserScan(project=False)
    label.open_scan(scan_file)
    label.open_label(label_file)
    u_label_sem = remap_lut[label.sem_label]
    if FLAGS.limit is not None:
      u_label_sem = u_label_sem[:FLAGS.limit]
      pts = pts[:FLAGS.limit]

    pred = SemLaserScan(project=False)
    pred.open_scan(scan_file)
    pred.open_label(pred_file)
    u_pred_sem = remap_lut[pred.sem_label]
    if FLAGS.limit is not None:
      u_pred_sem = u_pred_sem[:FLAGS.limit]

    dist = np.linalg.norm(pts, axis=1)

    evaluator_all.addBatch(u_pred_sem, u_label_sem)

    for start, end in range_list:
        if end == 10:
            mask = (dist >= start) & (dist <= end)
        else:
            mask = (dist >= start) & (dist < end)
        if np.any(mask):
            eval_dict[(start, end)].addBatch(u_pred_sem[mask], u_label_sem[mask])

  def print_metrics(eval_obj, name):
    print(f"\n========== {name} ==========")
    m_acc = eval_obj.getacc()
    m_iou, cls_iou = eval_obj.getIoU()
    m_prec, m_rec, m_f1, cls_prec, cls_rec, cls_f1 = eval_obj.get_precision_recall_f1()

    print(f'Acc avg:        {m_acc:.3f}')
    print(f'IoU avg:        {m_iou:.3f}')
    print(f'Precision avg:  {m_prec:.3f}')
    print(f'Recall avg:     {m_rec:.3f}')
    print(f'F1-score avg:   {m_f1:.3f}')

    for i, jacc in enumerate(cls_iou):
      if i not in ignore:
        print('Class {} [{}]: IoU = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
            i, class_strings[class_inv_remap[i]],
            jacc, cls_prec[i], cls_rec[i], cls_f1[i]))

  print_metrics(evaluator_all, "All (0~10m total)")

  for start, end in range_list:
      name = f"Range {start}~{end}m"
      print_metrics(eval_dict[(start, end)], name)

  print("\n" + "*" * 80)
  print("Done!")