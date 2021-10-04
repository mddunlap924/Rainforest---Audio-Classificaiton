import pickle
import os
import numpy as np
import tensorflow as tf
from lwlrap import LWLRAP
from HPTuning import model_setup_sed as model_setup
from HPTuning import preprocess_dataset_sed as preprocess_dataset
import pandas as pd
import tensorflow_addons as tfa
import gc


# Load training and testing datafiles in this directory
def load_datasets(data_folder):
    files_in_directory = os.listdir(data_folder)
    for file in files_in_directory:
        if 'training.pickle' in file:
            file_name_training = file
        if 'testing' in file:
            file_name_testing = file

    train_data_path = os.path.join(data_folder, file_name_training)
    with open(train_data_path, 'rb') as input_file:
        train = pickle.load(input_file)

    test_data_path = os.path.join(data_folder, file_name_testing)
    with open(test_data_path, 'rb') as input_file:
        test = pickle.load(input_file)
    return train, test


# Load saved pickle file with variables
def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as input_file:
        variables = pickle.load(input_file)
    train_idx = variables['output']['train_idx']
    val_idx = variables['output']['val_idx']
    folds = variables['hp_model'].folds
    model_inputs = variables['hp_model']

    return val_idx, model_inputs


# Load Tensorflow model with weights
def load_tf_model(inputs, model_name):
    model_weight_path = os.path.join(os.path.join(inputs.save_path, model_name), model_name + '.ckpt')
    if inputs.sed:
        model = model_setup.sed_model(inputs)
    else:
        model = model_setup.no_sed_model(inputs)
    model.load_weights(model_weight_path)
    return model


# Method to make several different score metrics
def several_metrics(truth, prediction):
    # LWLRAP
    # Performance metric
    lwlrap_metric = LWLRAP(24)
    lwlrap_score = lwlrap_metric(truth, prediction).numpy()

    # Micro-F1
    f1_micro = tfa.metrics.F1Score(num_classes=24, average='micro')
    f1_micro_score = f1_micro(truth, prediction).numpy()

    # Macro-F1
    f1_macro = tfa.metrics.F1Score(num_classes=24, average='macro')
    f1_macro_score = f1_macro(truth, prediction).numpy()

    # Precision
    precision = tf.keras.metrics.Precision()
    precision_score = precision(truth, prediction).numpy()

    # Recall
    recall = tf.keras.metrics.Recall()
    recall_score = recall(truth, prediction).numpy()

    scores = {'lwlrap': lwlrap_score,
              'f1_micro': f1_micro_score,
              'f1_macro': f1_macro_score,
              'precision': precision_score,
              'recall': recall_score
              }

    return scores


# Inference for a single model
def single_inference(tfmodel, model_input, model_name, val_x, val_y, test_x):
    # Build empty matrices for later storage
    if model_input.sed:
        test_predict = {'frame': np.zeros([1992, 24], dtype=np.float32),
                        'clip': np.zeros([1992, 24], dtype=np.float32),
                        'avg': np.zeros([1992, 24], dtype=np.float32),
                        'file_names': np.empty([1992], dtype='U14')
                        }
    else:
        test_predict = {'frame': np.zeros([1992, 24], dtype=np.float32),
                        'file_names': np.empty([1992], dtype='U14'),
                        }

    # Validation Data Inference for a single model
    val_set = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_set = val_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, 42, model_input,
                                                                             training=False))
    val_x = np.array([x.numpy() for x, _ in val_set])
    val_y = np.array([y.numpy() for _, y in val_set])
    val_y = np.max(val_y, axis=-2)
    val_pred_partial = tfmodel.predict(val_x)
    if model_input.sed:
        val_frame_pred = np.max(np.clip(val_pred_partial[0], 0.0, 1.0), axis=-2)
        val_clip_pred = np.clip(val_pred_partial[1], 0.0, 1.0)
        val_avg_pred = (val_frame_pred + val_clip_pred) / 2
        val_frame_score = several_metrics(val_y, val_frame_pred)
        val_clip_score = several_metrics(val_y, val_clip_pred)
        val_avg_score = several_metrics(val_y, val_avg_pred)

        val_scores_single_model = {'frame': val_frame_score,
                                   'clip': val_clip_score,
                                   'avg': val_avg_score,
                                   'model': model_name}
    else:
        val_frame_pred = val_pred_partial
        val_frame_score = several_metrics(val_y, val_frame_pred)
        val_scores_single_model = {'frame': val_frame_score, 'model': model_name}

    del val_set, val_x, val_y

    """ Test Data Inference for a Single Model """
    if model_input.sed:
        test_pred = {
            'frame': np.zeros([len(test_x), test_x.shape[1], 24], dtype=np.float32),
            'clip': np.zeros([len(test_x), test_x.shape[1], 24], dtype=np.float32),
            'avg': np.zeros([len(test_x), test_x.shape[1], 24], dtype=np.float32)
        }
    else:
        test_pred = {'frame': np.zeros([len(test_x), test_x.shape[1], 24], dtype=np.float32)}

    # Loop through test indexs b/c GPU cannot hold everything in memory
    for idx in range(test_x.shape[1]):
        x_sample_test = test_x[:, idx, :, :]
        y_sample_test = np.zeros([len(x_sample_test), 24, ])
        test_set = tf.data.Dataset.from_tensor_slices((x_sample_test, y_sample_test))
        test_set = test_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, 42, model_input,
                                                                                   training=False))
        test_x_partial = np.array([x.numpy() for x, _ in test_set])
        test_predict_partial = tfmodel.predict(test_x_partial)
        if model_input.sed:
            test_pred['frame'][:, idx, :] = np.max(np.clip(test_predict_partial[0], 0.0, 1.0), axis=-2)
            test_pred['clip'][:, idx, :] = np.clip(test_predict_partial[1], 0.0, 1.0)
            test_pred['avg'][:, idx, :] = (np.max(np.clip(test_predict_partial[0], 0.0, 1.0), axis=-2) +
                                           np.clip(test_predict_partial[1], 0.0, 1.0)) / 2
        else:
            test_pred['frame'][:, idx, :] = test_predict_partial
        print(f'Test Prediction for Count {idx + 1} of {test_x.shape[1]}')
        del x_sample_test, y_sample_test, test_set, test_x_partial, test_predict_partial
        gc.collect()

    # # Reshape test prediction back into number of samples
    # Insert prediction for the partial amount of data into the empty test_pred
    if model_input.sed:
        output_types = ['frame', 'clip', 'avg']
    else:
        output_types = ['frame']

    for out_type in output_types:
        test_predict[out_type] = test_pred[out_type].max(axis=1)

    print(f'Test Prediction Complete: Model {model_name}')
    del test_pred
    gc.collect()
    return test_predict, val_scores_single_model


# Create submission file
def create_submission(y, out_type, save_folder, file_names):
    sample_sub_path = r'C:\Kaggle\RainForest_R0\rfcx-species-audio-detection\sample_submission.csv'
    sample_sub = pd.read_csv(sample_sub_path)
    file_names = [file_name[:-5] for file_name in file_names]
    recording_ids = sample_sub['recording_id'].tolist()
    submission_df = sample_sub.copy()

    if file_names == recording_ids:
        submission_df.iloc[:, 1:] = y

    # Create Folder for Each Submission File
    folder_path = os.path.join(save_folder, out_type)
    if os.path.exists(os.path.join(save_folder, out_type)):
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)
    submission_df.to_csv(os.path.join(folder_path, 'submission.csv'), index=False)
    print(f'Submission File Created: {save_folder}')


# Return list of saved models in directory (these will be used for inference)
def list_all_models(variable):
    model_names = []
    for file in os.listdir(variable['hp_model'].save_path):
        if 'Fold' in file:
            if 'png' not in file:
                model_names.append(file)

    return model_names


def sed_inference(variables):
    """" User Inputs """
    DATA_FOLDER = os.path.join(r'C:\Kaggle\RainForest_R0\Datasets', 'Mel_' + variables['hp_model'].dataset)
    MODEL = variables['hp_model'].model
    PICKLE_PATH = os.path.join(variables['hp_model'].save_path,
                               variables['hp_model'].model + '_' + str(variables['hp_model'].iteration_number) +
                               '.pickle')

    """ Load and Preprocess Test Data """
    LWLRAP(24)
    train, test_class = load_datasets(DATA_FOLDER)
    test = test_class[0]
    del test_class
    gc.collect()

    """ List of test file names """
    test_file_names = test.file_names

    """ Generate output types based on model selection (SED or No SED) """
    if variables['hp_model'].sed:
        list_output_types = ['frame', 'clip', 'avg']
    else:
        list_output_types = ['frame']

    """ List of Models to Use for Inference """
    list_of_models = list_all_models(variables)

    """ Test Matrix Padding """
    if variables['hp_model'].sed:
        test_prediction = {'frame': np.zeros([1992, 24], dtype=np.float32),
                           'clip': np.zeros([1992, 24], dtype=np.float32),
                           'avg': np.zeros([1992, 24], dtype=np.float32),
                           'file_names': test_file_names
                           }
    else:
        test_prediction = {'frame': np.zeros([1992, 24], dtype=np.float32),
                           'file_names': test_file_names
                           }

    """ Start Inference Loop and Predict for Each Fold/Validation Model """
    val_idx, model_inputs = load_pickle(PICKLE_PATH)
    val_scores = []
    for i, model_name in enumerate(list_of_models):
        print(f'Inference for Model: {model_name}')
        model = load_tf_model(model_inputs, model_name)

        # Inferences on Validation and Test Data for a SINGLE Model
        test_predict_single_model, val_score = single_inference(model,
                                                                model_inputs,
                                                                model_name,
                                                                train.X[val_idx[i]],
                                                                train.Y[val_idx[i]],
                                                                test.X
                                                                )
        for output_type in list_output_types:
            test_prediction[output_type] += test_predict_single_model[output_type]
        val_scores.append(val_score)
        del model

    """ Average Results for submission"""
    submission_predictions = {}
    for output_type in list_output_types:
        test_prediction[output_type] = test_prediction[output_type] / len(list_of_models)

    # Create submission files
    for output_type in list_output_types:
        create_submission(test_prediction[output_type],
                          output_type,
                          model_inputs.save_path,
                          test_prediction['file_names'])

    # Save validation scores to folder for later analysis/lookup
    val_score_save_name = MODEL + '_' + str(model_inputs.iteration_number) + '_val_scores' + '.pickle'
    val_score_save_path = os.path.join(model_inputs.save_path, val_score_save_name)
    with open(val_score_save_path, 'wb') as output:
        pickle.dump(val_scores, output, pickle.HIGHEST_PROTOCOL)

    """ Print Validation Scores for All Models """
    lwlrap_avg = []
    f1_micro_avg = []
    for val_info in val_scores:
        model_val_score = val_info['model']
        lwlrap_val = val_info['frame']['lwlrap']
        f1_micro_val = val_info['frame']['f1_micro']
        print(f'{model_val_score} - LWLRAP {lwlrap_val}; F1 Mirco {f1_micro_val}')
        lwlrap_avg.append(lwlrap_val)
        f1_micro_avg.append(f1_micro_val)
    print(f'{model_name} Avg. LWLRAP Score: {sum(lwlrap_avg) / len(lwlrap_avg)}'
          f'Avg. F1 Micro Score: {sum(f1_micro_avg) / len(f1_micro_avg)} "\n"" ')

    print(f'Completed Inference for {model_name}')
