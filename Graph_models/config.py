GRAD_CLIP = 1.0

model_kwargs = {'num_nodes': 23,
                'num_flows': 529,
                'dropout': 0.5,
                'do_graph_conv': True,
                'addaptadj': True,
                'in_dim': 1,
                'apt_size': 10,
                'out_seq_len': 1,
                'in_seq_len': 6,
                'hidden': 32,
                'stride': 2,
                'kernel_size': 2,
                'blocks': 4,
                'layers': 2,
                'cat_feat_gc': False,
                'verbose': False,
                'device': 'cuda:0',
                'model': 'gwn'}

train_kwargs = {'lrate': 1e-4,
                'wdecay': 0.0001,
                'lr_decay_rate': 0.97,
                'epochs': 1000,
                'patience': 50,
                'batch_size': 64,
                'logdir': 'gwn',
                'dataset': 'sdn'}

is_train = True
DATASETS = ['geant', 'sdn']
MODELS = ['gwn', 'dcrnn']

# -----------------------------------------
train_kwargs['dataset'] = DATASETS[1]
model_kwargs['in_seq_len'] = 15
CUDA_DEVICE = 0
model_id = 1
# -----------------------------------------

model_kwargs['model'] = MODELS[model_id]
train_kwargs['logdir'] = model_kwargs['model']
model_kwargs['device'] = 'cuda:{}'.format(CUDA_DEVICE)
if 'sdn' in train_kwargs['dataset']:
    model_kwargs['num_nodes'] = 14
    model_kwargs['num_flows'] = 196

in_seq_len = model_kwargs['in_seq_len']

if in_seq_len == 3:
    model_kwargs['blocks'] = 1
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 2
    model_kwargs['stride'] = 2
elif in_seq_len == 6:
    model_kwargs['blocks'] = 2
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 2
    model_kwargs['stride'] = 2
elif in_seq_len == 12:
    model_kwargs['blocks'] = 4
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 2
    model_kwargs['stride'] = 2
elif in_seq_len == 15:
    model_kwargs['blocks'] = 2
    model_kwargs['layers'] = 3
    model_kwargs['kernel_size'] = 2
    model_kwargs['stride'] = 2
elif in_seq_len == 24:
    model_kwargs['blocks'] = 4
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 3
    model_kwargs['stride'] = 3
elif in_seq_len == 30:
    model_kwargs['blocks'] = 4
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 3
    model_kwargs['stride'] = 3
elif in_seq_len == 60:
    model_kwargs['blocks'] = 6
    model_kwargs['layers'] = 2
    model_kwargs['kernel_size'] = 4
    model_kwargs['stride'] = 4
else:
    raise NotImplemented('Not supported!')
