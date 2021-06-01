# Note: #SpecAug can be [None,None] or [int, int], it cannot be [None, int] or [int, None] (line 58 on preprocess)
# Note: layer_combination - 'Avg', 'Max', 'AvgMax'
# Note: model - 'ResNet18', 'ResNet34', 'EfficientNetB0', 'EfficientNetB2', 'MobileNetV2'

# 'output_bias': [-4.5],

def get_inputs():
    inputs = {
        'dataset': ['224_512_1'],
        'model': ['EfficientNetB0'],
        'learning_rate_min': [5e-6],
        'learning_rate_max': [1e-3],
        'layer_combination': ['Avg'],
        'regularization': [0.001, 0.005],
        'dropout0': [0.4],
        'num_hidden': [256],
        'dropout1': [0.3],
        'patience': [100],
        'clr_step': [8],
        'output_bias': [None],
        'folds': [5],
        'batch_size': [12],
        'training': [{'Gaussian': None, 'Brightness': None, 'SpecAug': [None, None]}],
        'clr_mode': ['triangular'],
        'mixup': [0.4]
    }

    count = 0
    model_combinations = {}
    for dataset in inputs['dataset']:
        for model in inputs['model']:
            for learning_rate_min in inputs['learning_rate_min']:
                for learning_rate_max in inputs['learning_rate_max']:
                    for layer_combination in inputs['layer_combination']:
                        for regularization in inputs['regularization']:
                            for dropout0 in inputs['dropout0']:
                                for num_hidden in inputs['num_hidden']:
                                    for dropout1 in inputs['dropout1']:
                                        for patience in inputs['patience']:
                                            for clr_step in inputs['clr_step']:
                                                for output_bias in inputs['output_bias']:
                                                    for folds in inputs['folds']:
                                                        for batch_size in inputs['batch_size']:
                                                            for training in inputs['training']:
                                                                for clr_mode in inputs['clr_mode']:
                                                                    for mixup in inputs['mixup']:
                                                                        model_combinations[count] = {'dataset': dataset,
                                                                                                     'model': model,
                                                                                                     'learning_rate_min': learning_rate_min,
                                                                                                     'learning_rate_max': learning_rate_max,
                                                                                                     'layer_combination': layer_combination,
                                                                                                     'regularization': regularization,
                                                                                                     'dropout0': dropout0,
                                                                                                     'num_hidden': num_hidden,
                                                                                                     'dropout1': dropout1,
                                                                                                     'patience': patience,
                                                                                                     'clr_step': clr_step,
                                                                                                     'output_bias': output_bias,
                                                                                                     'folds': folds,
                                                                                                     'batch_size': batch_size,
                                                                                                     'training': training,
                                                                                                     'clr_mode':clr_mode,
                                                                                                     'mixup': mixup}
                                                                        count += 1
    return model_combinations

# print('End of Script')
