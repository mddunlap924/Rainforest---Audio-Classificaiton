import pickle
import os
import random
import numpy as np
import misc_methods
import matplotlib.pyplot as plt
from dataset import Dataset
import tensorflow as tf
import HPTuning.HP_Info as HP_Info
import HPTuning.model_training as model_training
from lwlrap import LWLRAP

def model_score(results, y_true, idxs):
    y_pred = np.empty([y_true.shape[0], y_true.shape[1]], dtype=float)
    for i, idx in enumerate(idxs):
        y_pred[idx] = results[i]['y_predict']
    lwlrap_metric = LWLRAP(24)
    lwlrap_score = lwlrap_metric(y_true, y_pred).numpy()

    return lwlrap_score


# Save python variables to cwd with given filename
def save_pickle(save_folder, file_name, variables):
    save_path = os.path.join(save_folder, file_name + '.pickle')
    with open(save_path, 'wb') as output:
        pickle.dump(variables, output, pickle.HIGHEST_PROTOCOL)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
SEED = 42

""" Load Model Inputs """
hp_models = HP_Info.hp_combinations_to_run()

""" Loop HP Models into Training """
model_results = []
for hp_model in hp_models:
    output = model_training.train_model(hp_model, SEED)
    lwlrap_score = model_score(output['results'], output['y_true'], output['val_idx'])
    output['lwlrap_score'] = lwlrap_score

    # Save variables for later lookup and analysis
    save_variables = {'hp_model': hp_model,
                      'output': output
                      }

    save_pickle(hp_model.save_path, hp_model.save_path.split('\\')[-1], save_variables)
    model_results.append([hp_model.save_path.split('\\')[-1], lwlrap_score])

    for i, model_result in enumerate(model_results):
        print(f'Model {i} of {len(hp_models)} - {model_result[0]} - LWLRAP: {model_result[1]}')

    del output

    #TODO add a check so an inference submission file be created if its above certain score



print('End to Training')
