GRAD_CLIP = 1.0

model_kwargs = {'num_nodes': 529,
                'dropout': 0.5,
                'do_graph_conv': True,
                'addaptadj': True,
                'in_dim': 1,
                'apt_size': 10,
                'out_seq_len': 1,
                'hidden': 32,
                'stride': 2,
                'kernel_size': 2,
                'blocks': 4,
                'layers': 2,
                'cat_feat_gc': False,
                'verbose': False,
                'device': 'cuda:0'}

train_kwargs = {'lrate': 1e-4,
                'wdecay': 0.0001,
                'lr_decay_rate': 0.97,
                'epochs': 1000,
                'patience': 50,
                'batch_size': 64}

is_train = False