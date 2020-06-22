from collections import defaultdict, deque
import pickle
from attrdict import AttrDict
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
np.seterr(invalid="ignore")

xrange = range

class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)
        
    
    def save(self, model, optimizer, epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        filename = os.path.join(self.path, 'model_{:05}.pth'.format(epoch))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if os.path.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
            
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
    
    def load(self, model, optimizer):
        """
        Return starting epoch
        """
        with open(self.listfile, 'rb') as f:
            model_list = pickle.load(f)
            if len(model_list) == 0:
                print('No checkpoint found. Starting from scratch')
                return 0
            else:
                checkpoint = torch.load(model_list[-1])
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('Load checkpoint from {}.'.format(model_list[-1]))
                return checkpoint['epoch']


def get_scores_and_plot(scorer,
                        data_abs_xy,
                        activations,
                        directory,
                        filename,
                        plot_graphs=True,  # pylint: disable=unused-argument
                        nbins=20,  # pylint: disable=unused-argument
                        cm="jet",
                        sort_by_score_60=True):
  """Plotting function."""

  # Concatenate all trajectories
  xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
  act = activations.reshape(-1, activations.shape[-1])
  n_units = act.shape[1]
  # Get the rate-map for each unit
  s = [
      scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
      for i in range(n_units)
  ]
  # Get the scores
  score_60, score_90, max_60_mask, max_90_mask, sac = zip(
      *[scorer.get_scores(rate_map) for rate_map in s])
  # Separations
  # separations = map(np.mean, max_60_mask)
  # Sort by score if desired
  if sort_by_score_60:
    ordering = np.argsort(-np.array(score_60))
  else:
    ordering = range(n_units)
  # Plot
  cols = 16
  rows = int(np.ceil(n_units / cols))
  fig = plt.figure(figsize=(24, rows * 4))
  for i in xrange(n_units):
    rf = plt.subplot(rows * 2, cols, i + 1)
    acr = plt.subplot(rows * 2, cols, n_units + i + 1)
    if i < n_units:
      index = ordering[i]
      title = "%d (%.2f)" % (index, score_60[index])
      # Plot the activation maps
      scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
      # Plot the autocorrelation of the activation maps
      scorer.plot_sac(
          sac[index],
          mask_params=max_60_mask[index],
          ax=acr,
          title=title,
          cmap=cm)
  # Save
  if not os.path.exists(directory):
    os.makedirs(directory)
  with PdfPages(os.path.join(directory, filename), "w") as f:
    plt.savefig(f, format="pdf")
  plt.close(fig)
  return (np.asarray(score_60), np.asarray(score_90),
          np.asarray(map(np.mean, max_60_mask)),
          np.asarray(map(np.mean, max_90_mask)))