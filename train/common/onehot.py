# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger


class oneHot(nn.Module):
  def __init__(self, device, nclasses, spatial_dim=2):
    super().__init__()
    self.device = device
    self.nclasses = nclasses
    self.spatial_dim = spatial_dim

  def onehot1dspatial(self, x):
    assert(len(x.shape) == 1 or len(x.shape) == 2)
    remove_dim = False
    if len(x.shape) == 1:
      x = x[None, ...]
      remove_dim = True

    n, b = x.shape

    one_hot = torch.zeros((n, self.nclasses, b),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def onehot2dspatial(self, x):
    assert(len(x.shape) == 2 or len(x.shape) == 3)
    remove_dim = False
    if len(x.shape) == 2:
      x = x[None, ...]
      remove_dim = True

    n, h, w = x.shape

    one_hot = torch.zeros((n, self.nclasses, h, w),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def forward(self, x):
    if self.spatial_dim == 1:
      return self.onehot1dspatial(x)
    elif self.spatial_dim == 2:
      return self.onehot2dspatial(x)


if __name__ == "__main__":
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  nclasses = 6
  print("*"*80)
  print("Num classes 1d =", nclasses)
  print("*"*80)

  print("Tensor 1d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=1)
  print("out:", module(tensor))
  print("*"*80)

  print("*"*80)
  print("Tensor 1d spat dim, batched")
  tensor = torch.arange(0, nclasses).to(device)
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=1)
  print("out:", module(tensor))
  print("*"*80)

  nclasses = 3
  print("*"*80)
  print("Num classes 2d =", nclasses)
  print("*"*80)

  print("*"*80)
  print("Tensor 2d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0),
                      tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=2)
  print("out:", module(tensor))
  print("*"*80)

  print("*"*80)
  print("Tensor 2d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0),
                      tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=2)
  print("out:", module(tensor))
  print("*"*80)
