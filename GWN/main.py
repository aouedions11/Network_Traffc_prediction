import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn
from keras.layers import LSTM, Bidirectional
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tqdm import tqdm
import time
import torch.optim as optim
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# Building Model
from keras.layers.core import Dense, Dropout
import os
from data_loading import *
from gwn import GWNet
from config import model_kwargs, GRAD_CLIP, train_kwargs, is_train
from utils import calc_metrics
from logger import Logger
lossfn = torch.nn.MSELoss()


def training(model, optimizer, x, y, scaler=None):
    model.train()
    optimizer.zero_grad()
    # input = torch.nn.functional.pad(input, (1, 0, 0, 0))

    output = model(x)  # now, output = [bs, seq_y, n]
    if scaler is not None:
        predict = scaler.inverse_transform(output)
    else:
        predict = output

    if len(predict.size()) != len(y.size()):
        predict = torch.reshape(predict, y.shape)

    loss = lossfn(predict, y)
    rse, mae, mse, mape, rmse = calc_metrics(predict, y)
    loss.backward()

    if GRAD_CLIP is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()


def evaluating(model, x, y, scaler=None):

    model.eval()

    output = model(x)  # now, output = [bs, seq_y, n]

    if scaler is not None:
        predict = scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=10e10)
    else:
        predict = output

    loss = lossfn(predict, y)
    rse, mae, mse, mape, rmse = calc_metrics(predict, y)

    return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()


def testing(model, test_loader, out_seq_len, scaler=None):
    model.eval()
    outputs = []
    y_real = []
    for _, batch in enumerate(test_loader):
        x = batch['x']  # [b, seq_x, n, f]
        y = batch['y']  # [b, seq_y, n]

        preds = model(x)
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            preds = torch.clamp(preds, min=0., max=10e10)
        else:
            preds = preds

        outputs.append(preds)
        y_real.append(y)

    yhat = torch.cat(outputs, dim=0)
    y_real = torch.cat(y_real, dim=0)
    test_met = []

    yhat[yhat < 0.0] = 0.0

    if len(yhat.size()) != len(y_real.size()):
        yhat = torch.reshape(yhat, y_real.shape)

    test_met.append([x.item() for x in calc_metrics(yhat, y_real)])
    test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, y_real, yhat


DEVICE = 'cuda:0'
device = torch.device(DEVICE)

train_loader, val_loader, test_loader, scaler = get_dataloader(device)

aptinit = None
supports = None

model = GWNet.from_args(supports, aptinit, **model_kwargs)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=train_kwargs['lrate'], weight_decay=train_kwargs['wdecay'])
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: train_kwargs['lr_decay_rate'] ** epoch)

logger = Logger()

if is_train:
    print('|--- Training ---|')
    iterator = trange(train_kwargs['epochs'])
    tmps_train = time.time()
    for epoch in iterator:
        train_loss, train_rse, train_mae, train_mse, train_mape, train_rmse = [], [], [], [], [], []
        for iter, batch in enumerate(train_loader):
            x = batch['x']  # [b, seq_x, n, f]
            y = batch['y']  # [b, seq_y, n]
            if y.max() == 0: continue

            loss, rse, mae, mse, mape, rmse = training(model, optimizer, x, y, scaler=None)

            train_loss.append(loss)
            train_rse.append(rse)
            train_mae.append(mae)
            train_mse.append(mse)
            train_mape.append(mape)
            train_rmse.append(rmse)

        scheduler.step()

        with torch.no_grad():
            val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = [], [], [], [], [], []
            for _, batch in enumerate(val_loader):
                x = batch['x']  # [b, seq_x, n, f]
                y = batch['y']  # [b, seq_y, n]
                metrics = evaluating(model, x, y, scaler=None)
                val_loss.append(metrics[0])
                val_rse.append(metrics[1])
                val_mae.append(metrics[2])
                val_mse.append(metrics[3])
                val_mape.append(metrics[4])
                val_rmse.append(metrics[5])

        m = dict(train_loss=np.mean(train_loss), train_rse=np.mean(train_rse),
                 train_mae=np.mean(train_mae), train_mse=np.mean(train_mse),
                 train_mape=np.mean(train_mape), train_rmse=np.mean(train_rmse),
                 val_loss=np.mean(val_loss), val_rse=np.mean(val_rse),
                 val_mae=np.mean(val_mae), val_mse=np.mean(val_mse),
                 val_mape=np.mean(val_mape), val_rmse=np.mean(val_rmse))

        description = logger.summary(m, model, epoch, train_kwargs['patience'])

        if logger.stop:
            break

        description = 'Epoch: {} '.format(epoch) + description
        iterator.set_description(description)

    tmps_t2 = time.time() - tmps_train
    print("Train_Temps = %f" % tmps_t2)

model.load_state_dict(torch.load(logger.best_model_save_path))

print('|--- Testing ---|')
with torch.no_grad():
    test_met_df, y_real, yhat = testing(model, test_loader, out_seq_len=1, scaler=None)
print('Test metric: ', test_met_df)
