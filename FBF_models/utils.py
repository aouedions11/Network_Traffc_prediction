import torch
from scipy.io import loadmat
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg


EPS = 1e-8


def rse(preds, labels):
    return torch.sum((preds - labels) ** 2) / torch.sum((labels + EPS) ** 2)


def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def mse(preds, labels):
    return torch.mean((preds - labels) ** 2)


def mape(preds, labels):
    return torch.mean(torch.abs((preds - labels) / (labels + EPS)))


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def calc_metrics(preds, labels):
    return rse(preds, labels), mae(preds, labels), mse(preds, labels), mape(preds, labels), rmse(preds, labels)
