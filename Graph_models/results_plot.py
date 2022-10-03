from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import os

parent_log = '../logs'

DATASETS = ['geant', 'sdn']
IN_SEQ_LEN = [[3, 6, 12, 24], [15, 30, 60]]

dataset = 1
model = 'gwn'
NRUN = 5
results = []


for seqlen in IN_SEQ_LEN[dataset]:
    for run in range(NRUN):
        path = os.path.join(parent_log, '{}_data_{}_seq_{}/run_{}'.format(model, DATASETS[dataset], seqlen, run))
        file_name = os.path.join(path, 'test_metrics.csv')

        if os.path.isfile(file_name):
            result = np.loadtxt(file_name, skiprows=1, delimiter=',')
        else:
            result = np.zeros(shape=(6,))

        result[0] = seqlen
        results.append(result)

results = np.stack(results)
print(results)
print(results.shape)
results_df = pd.DataFrame(results, columns=['seq_len', 'rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('run')

results_df.to_csv(os.path.join(parent_log, 'results_{}_data_{}.csv'.format(model, DATASETS[dataset])))

print('Average results:\n')
for i in range(0, results.shape[0], NRUN):

    avg_results = np.mean(results[i:i+NRUN], axis=0)
    print(avg_results[1:])