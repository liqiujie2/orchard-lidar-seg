#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import numpy as np
import time

class iouEval:
  def __init__(self, n_classes, device, ignore=None):
    self.n_classes = n_classes
    self.device = device
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor(
        [n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()
    self.ones = None
    self.last_scan_size = None

  def addBatch(self, x, y):
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(np.array(x)).long().to(self.device)
    if isinstance(y, np.ndarray):
      y = torch.from_numpy(np.array(y)).long().to(self.device)

    x_row = x.reshape(-1)
    y_row = y.reshape(-1)

    idxs = torch.stack([x_row, y_row], dim=0)

    if self.ones is None or self.last_scan_size != idxs.shape[-1]:
      self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
      self.last_scan_size = idxs.shape[-1]
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), self.ones, accumulate=True)

  def getStats(self):
    conf = self.conf_matrix.clone().double()
    conf[self.ignore] = 0
    conf[:, self.ignore] = 0

    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    union = tp + fp + fn + 1e-15
    iou = tp / union
    iou_mean = iou[self.include].mean()
    return iou_mean, iou

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean

  def get_precision_recall_f1(self):
    tp, fp, fn = self.getStats()
    epsilon = 1e-15

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1[torch.isnan(f1)] = 0

    precision_mean = precision[self.include].mean()
    recall_mean = recall[self.include].mean()
    f1_mean = f1[self.include].mean()

    return precision_mean, recall_mean, f1_mean, precision, recall, f1

if __name__ == "__main__":
  nclasses = 2
  ignore = []

  lbl = torch.zeros((7, 7)).long()
  argmax = torch.zeros((7, 7)).long()
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  eval = iouEval(nclasses, torch.device('cpu'), ignore)

  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  m_acc = eval.getacc()
  m_prec, m_recall, m_f1, _, _, _ = eval.get_precision_recall_f1()

  print("*" * 80)
  print("Small iou mock problem")
  print("IoU mean:       ", m_iou)
  print("IoU per class:  ", iou)
  print("Acc:            ", m_acc)
  print("Precision mean: ", m_prec)
  print("Recall mean:    ", m_recall)
  print("F1-score mean:  ", m_f1)
  print("*" * 80)
