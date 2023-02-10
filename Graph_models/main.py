import torch.nn
# Building Model
import os
import time

import torch.nn
import torch.optim as optim
from tqdm import trange

from config import model_kwargs, GRAD_CLIP, is_train
from data_loading import *
from dcrnn import DCRNNModel
from gwn import GWNet
from logger import Logger
from utils import calc_metrics
from lstm import LSTM
lossfn = torch.nn.MSELoss()


def training(model, optimizer, x, y, scaler=None):
    model.train()
    optimizer.zero_grad()
    # input = torch.nn.functional.pad(input, (1, 0, 0, 0))
    # print('x:', x.size())
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
    yreal = []
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
        yreal.append(y)

    yhat = torch.cat(outputs, dim=0)
    yreal = torch.cat(yreal, dim=0)
    test_met = []

    yhat[yhat < 0.0] = 0.0

    if len(yhat.size()) != len(yreal.size()):
        yhat = torch.reshape(yhat, yreal.shape)

    print(yhat.size())

    test_met.append([x.item() for x in calc_metrics(yhat, yreal)])

    test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse'])
    return test_met_df, yreal, yhat


print(model_kwargs)
print(model_kwargs)

device = torch.device(model_kwargs['device'])
dataset = model_kwargs['dataset']
seq_len = model_kwargs['in_seq_len']

train_loader, val_loader, test_loader, scaler = get_dataloader(**model_kwargs)

aptinit = None
supports = None

parent_logs_path = '../logs'
logdir = model_kwargs['logdir']
logdir = logdir + '_data_{}_seq_{}'.format(dataset, seq_len)
for run in range(0, 5, 1):
    _logdir = os.path.join(parent_logs_path, logdir, 'run_{}'.format(run))
    logger = Logger(logdir=_logdir)

    if 'dcrnn' in model_kwargs['model']:
        m_adj = np.load(os.path.join('../data', '{}_adj.npy'.format(dataset)))
        model = DCRNNModel(adj_mx=m_adj, seq_len=model_kwargs['in_seq_len'],
                           nodes=model_kwargs['num_nodes'],
                           pre_len=model_kwargs['out_seq_len'], device=device)
    elif 'lstm' in model_kwargs['model']:
        model = LSTM(in_dim=model_kwargs['num_flows'], hidden_dim=model_kwargs['hidden'], n_layer=3,
                     seq_len=model_kwargs['in_seq_len'], pre_len=model_kwargs['out_seq_len'])

    elif 'gwn' in model_kwargs['model']:
        model = GWNet.from_args(supports, aptinit, **model_kwargs)
    else:
        raise NotImplemented('Model is not supported!')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=model_kwargs['lrate'], weight_decay=model_kwargs['wdecay'])
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: model_kwargs['lr_decay_rate'] ** epoch)

    if is_train:
        print('|--- Training {} ---|'.format(run))
        iterator = trange(model_kwargs['epochs'])
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

            description = logger.summary(m, model, epoch, model_kwargs['patience'])

            if logger.stop:
                break

            description = 'Epoch: {} '.format(epoch) + description
            iterator.set_description(description)

        tmps_t2 = time.time() - tmps_train
        print("Train_Temps = %f" % tmps_t2)

    model.load_state_dict(torch.load(logger.best_model_save_path))

    print('|--- Testing {} ---|'.format(run))
    with torch.no_grad():
        test_met_df, yreal, yhat = testing(model, test_loader, out_seq_len=model_kwargs['out_seq_len'], scaler=None)

    test_met_df.to_csv(os.path.join(logger.logdir, 'test_metrics.csv'))
    np.save(os.path.join(logger.logdir, 'y_real_data'), yreal.detach().cpu().numpy())
    np.save(os.path.join(logger.logdir, 'y_hat_data'), yhat.detach().cpu().numpy())
    print('Test metric: ', test_met_df)
