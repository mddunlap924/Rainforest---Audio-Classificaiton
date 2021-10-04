# Note: #SpecAug can be [None,None] or [int, int], it cannot be [None, int] or [int, None] (line 58 on preprocess)
# Note: layer_combination - 'Avg', 'Max', 'AvgMax'
# Note: model - 'ResNet18', 'ResNet34', 'EfficientNetB0', 'EfficientNetB2', 'MobileNetV2'

def get_inputs():
    inputs = {
        'dataset': ['224_224_33', '224_224_53', '224_224_123'],
        # 'dataset': ['224_224_53'],
        'model': ['DenseNet121', 'EfficientNetB3'],
        'sed': [True],
        'base_trainable': [True],
        'augment': [True],
        'fp': [False],
        'drop_connect_rate': [0.2],
        'learning_rate_min': [5e-6],
        'learning_rate_max': [1e-3],
        'fc_layer': [{'dropout0': 0.4, 'num_hidden': 512, 'dropout1': 0.25},
                     ],
        'epochs': [75],
        'lr_schedule': [{'type': 'clr', 'clr_step': 4, 'mode': 'triangular'}],
        # 'lr_schedule': [{'type': 'sgd', 'lr_decay': 0.8, 'cycle_length': 4, 'mult_factor': 1.5}],
        'folds': [5],
        'batch_size': [16],
        'training': [{'Gaussian': None, 'Brightness': None, 'SpecAug': [None, None]}],
        'mixup': [None],
        'loss_weights': [[0.2, 0.8]],
        'bias_initializer': [-4.5]
    }

    count = 0
    model_combinations = {}
    for dataset in inputs['dataset']:
        for model in inputs['model']:
            for sed in inputs['sed']:
                for base_trainable in inputs['base_trainable']:
                    for augment in inputs['augment']:
                        for fp in inputs['fp']:
                            for drop_connect_rate in inputs['drop_connect_rate']:
                                for learning_rate_min in inputs['learning_rate_min']:
                                    for learning_rate_max in inputs['learning_rate_max']:
                                        for fc_layer in inputs['fc_layer']:
                                            for epochs in inputs['epochs']:
                                                for lr_schedule in inputs['lr_schedule']:
                                                    for folds in inputs['folds']:
                                                        for batch_size in inputs['batch_size']:
                                                            for training in inputs['training']:
                                                                for mixup in inputs['mixup']:
                                                                    for loss_weights in inputs['loss_weights']:
                                                                        for bias_initializer in inputs['bias_initializer']:
                                                                            model_combinations[count] = {'dataset': dataset,
                                                                                                         'model': model,
                                                                                                         'sed': sed,
                                                                                                         'base_trainable': base_trainable,
                                                                                                         'augment': augment,
                                                                                                         'fp': fp,
                                                                                                         'drop_connect_rate':
                                                                                                             drop_connect_rate,
                                                                                                         'learning_rate_min': learning_rate_min,
                                                                                                         'learning_rate_max': learning_rate_max,
                                                                                                         'fc_layer': fc_layer,
                                                                                                         'epochs': epochs,
                                                                                                         'lr_schedule': lr_schedule,
                                                                                                         'folds': folds,
                                                                                                         'batch_size': batch_size,
                                                                                                         'training': training,
                                                                                                         'mixup': mixup,
                                                                                                         'loss_weights':
                                                                                                             loss_weights,
                                                                                                         'bias_initializer':
                                                                                                             bias_initializer,
                                                                                                         }
                                                                            count += 1
    return model_combinations
