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
                'model': 'gwn',
                'batch_size': 64,
                'lrate': 1e-4,
                'wdecay': 0.0001,
                'lr_decay_rate': 0.97,
                'epochs': 1000,
                'patience': 50,
                'logdir': 'gwn',
                'dataset': 'sdn'}

DATASETS = ['geant', 'sdn', 'abilene']
MODELS = ['gwn', 'dcrnn', 'lstm']

# -----------------------------------------
dataset = 0
model_kwargs['in_seq_len'] = 24
cuda_device = 0
model_id = 1
is_train = True
# -----------------------------------------

model_kwargs['dataset'] = DATASETS[dataset]
model_kwargs['model'] = MODELS[model_id]
model_kwargs['logdir'] = MODELS[model_id]
model_kwargs['device'] = 'cuda:{}'.format(cuda_device)

if 'sdn' in model_kwargs['dataset']:
    model_kwargs['num_nodes'] = 14
    model_kwargs['num_flows'] = 196
elif 'abilene' in model_kwargs['dataset']:
    model_kwargs['num_nodes'] = 12
    model_kwargs['num_flows'] = 144
elif 'geant' in model_kwargs['dataset']:
    model_kwargs['num_nodes'] = 23
    model_kwargs['num_flows'] = 529
else:
    raise NotImplemented('Dataset is not supported!')

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

