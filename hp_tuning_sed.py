import pickle
import os
import numpy as np
import tensorflow as tf
import HPTuning.HP_Info_SED as HP_Info
import HPTuning.model_training_sed as model_training
import HPTuning.hp_inference_sed4 as inference
from lwlrap import LWLRAP
import email_results
import gc
from dataset3 import Dataset


def model_score(results, y_true, idxs):
    y_true_ = np.max(y_true, axis=-2)
    y_pred = np.empty([y_true.shape[0], y_true.shape[2]], dtype=float)
    for i, idx in enumerate(idxs):
        y_pred[idx] = np.max(results[i]['y_predict']['frame'], axis=-2)
    lwlrap_metric = LWLRAP(24)
    lwlrap_score = lwlrap_metric(y_true_, y_pred).numpy()

    return lwlrap_score

def model_score_no_sed(results, y_true, idxs):
    y_true_ = np.max(y_true, axis=-2)
    y_pred = np.empty([y_true.shape[0], y_true.shape[2]], dtype=float)
    for i, idx in enumerate(idxs):
        y_pred[idx] = results[i]['y_predict']['frame']
    lwlrap_metric = LWLRAP(24)
    lwlrap_score = lwlrap_metric(y_true_, y_pred).numpy()

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

# """ Loop HP Models into Training """
model_results = []
for hp_model in hp_models:
    """ Train Model """
    output = model_training.train_model(hp_model, SEED)
    if hp_model.sed:
        lwlrap_score = model_score(output['results'], output['y_true'], output['val_idx'])
    else:
        lwlrap_score = model_score_no_sed(output['results'], output['y_true'], output['val_idx'])
    output['lwlrap_score'] = lwlrap_score

    # Save variables for later lookup and analysis
    save_variables = {'hp_model': hp_model,
                      'output': output
                      }
    save_pickle(hp_model.save_path, hp_model.save_path.split('\\')[-1], save_variables)
    model_results.append([hp_model.save_path.split('\\')[-1], lwlrap_score])

    # Print Results to Screen
    for i, model_result in enumerate(model_results):
        print(f'Model {i + 1} of {len(hp_models)} - {model_result[0]} - Frame LWLRAP: {model_result[1]}')
    del output, save_variables
    gc.collect()

    """ Inference on Test Data with Trained Model """
    # file_name = r'C:\Kaggle\RainForest_R0\HPTuning\224_224_33\ResNet18_SED\ResNet18_0\ResNet18_0.pickle'
    file_name = hp_model.model + '_' + hp_model.save_path[-1] + '.pickle'
    with open(os.path.join(hp_model.save_path, file_name), 'rb') as input_file:
        save_variables = pickle.load(input_file)

    # Create Submission Files
    inference.sed_inference(save_variables)
    print(f'Completed Everything for Model {hp_model.save_path}')
    del save_variables, file_name
    gc.collect()

del model_result, model_results
gc.collect()


# """ INFERENCE """
# for i, hp_model in enumerate(hp_models):
#     """ To Only Make An Inference On Existing Dataset Load the Pickle File - Toggle/on/off for Code Editing """
#     # file = r'C:\Kaggle\RainForest_R0\HPTuning\224_512_1\ResNet18_SED\ResNet18_0\ResNet18_0.pickle'
#     file_name = hp_model.model + '_' + str(i) + '.pickle'
#     with open(os.path.join(hp_model.save_path, file_name), 'rb') as input_file:
#         save_variables = pickle.load(input_file)
#
#     # Create Submission Files
#     inference.sed_inference(save_variables)
#     print(f'Completed Everything for Model {hp_model.save_path}')
#     del save_variables
#     gc.collect()

""" Send Email with Summary of Results """
email_results.send_email()

print('End HP Tuning SED')
