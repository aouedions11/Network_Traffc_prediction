import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import os
import torch


def display_stats(stats):
    print(
        'epoch={} -- rse={:0.4f} | mae={:0.4f} | mape={:0.4f} | mse={:0.4f} | rmse={:0.4f}'. \
            format(int(stats['epoch']),
                   stats['rse/infer'],
                   stats['mae/infer'],
                   stats['mape/infer'],
                   stats['mse/infer'],
                   stats['rmse/infer']))


class Logger:

    def __init__(self):
        parent_logs_path = '../logs'
        log_dir = 'gwn'
        log_dir = os.path.join(parent_logs_path, log_dir)

        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        self.min_val_loss = np.inf
        self.patience = 0
        self.best_model_save_path = os.path.join(log_dir, 'best_model.pth')

        self.metrics = []
        self.stop = False

    def summary(self, m, model, epoch, patience):
        m = pd.Series(m)
        self.metrics.append(m)
        if m.val_loss < self.min_val_loss:
            torch.save(model.state_dict(), self.best_model_save_path)
            self.patience = 0
            self.min_val_loss = m.val_loss
        else:
            self.patience += 1
        met_df = pd.DataFrame(self.metrics)
        description = 'train loss: {:.5f} val_loss: {:.5f} | best val_loss: {:.5f} patience: {}'.format(
            m.train_loss,
            m.val_loss,
            self.min_val_loss,
            self.patience)

        met_df.round(6).to_csv('{}/train_metrics.csv'.format(self.log_dir))

        self.writer.add_scalar('Loss/Train', m.train_loss, epoch)
        self.writer.add_scalar('Loss/Val', m.val_loss, epoch)

        if self.patience >= patience:
            self.stop = True
        return description

