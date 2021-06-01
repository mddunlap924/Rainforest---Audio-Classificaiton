import pickle
import os
import random
import numpy as np
import misc_methods
import matplotlib.pyplot as plt
from dataset import Dataset
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from lwlrap import LWLRAP
from HPTuning import model_setup
from HPTuning import preprocess_dataset
import time
import pandas as pd


# Load training and testing datafiles in this directory
def load_datasets(data_folder):
    files_in_directory = os.listdir(data_folder)
    for file in files_in_directory:
        if 'training' in file:
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
def load_pickle(data_folder, model_folder, fold_name):
    file_path = os.path.join(os.path.join(data_folder, model_folder), fold_name)
    with open(os.path.join(file_path, 'save_variables.pickle'), 'rb') as input_file:
        variables = pickle.load(input_file)

    train_idx = variables['train_idx']
    val_idx = variables['val_idx']
    folds = variables['folds']
    output_bias = variables['output_bias']

    return output_bias, val_idx


# Return list of saved models in directory (these will be used for inference)
def list_of_models(data_folder, model_name):
    model_names = []
    for file in os.listdir(os.path.join(data_folder, model_name)):
        if 'Fold' in file:
            model_names.append(file)

    return model_names


# Load Tensorflow model with weights
def load_tf_model(data_folder, model_name, cp_folder, output_bias):
    model_weight_folder = os.path.join(os.path.join(data_folder, model_name), cp_folder)

    if 'ResNet34' in model_name:
        model = model_setup.resnet34(output_bias)

    # model.load_weights(os.path.join(model_weight_folder, cp_folder + '.ckpt')).expect_partial()
    model.load_weights(os.path.join(model_weight_folder, cp_folder + '.ckpt'))

    return model


# Preprocess Test TF Data
def test_dataset(test_x, num_windows):
    # Go from Test Class (created by Dunlap) to TF Dataset for preprocessing

    # TODO organize TF dataset because its normalized by 6 and causes errors

    test_predictions = np.zeros([len(test_x), 24])
    test_set = tf.data.Dataset.from_tensor_slices((test_x, test_predictions))
    test_set = test_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, seed=42, training=False))
    if CHECK_TEST_IMAGES:
        preprocess_dataset.check_test_image2(test_set, test_x, num_windows)

    # Prepare to convert from TF dataset into numpy array
    for i in test_set.take(1):
        test_set_shape = i[0].numpy().shape
    test_in_numpy = np.zeros([len(test_x), test_set_shape[0], test_set_shape[1], test_set_shape[2]])

    tic = time.perf_counter()
    count = 0
    for element in test_set.as_numpy_iterator():
        test_in_numpy[count, :, :, :] = element[0]
        count += 1
    print(f'Time from Test Class, TF Preprocess, to Numpy Array: {round(time.perf_counter() - tic, 2)} seconds')

    return test_in_numpy


# Inference for a single model
def inference(model, x_test, test_idx, train_X, train_Y):
    # Performance metric
    lwlrap_metric = LWLRAP(24)

    # Validation Data Inference
    tic = time.perf_counter()
    val_set = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    val_set = val_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, seed=42, training=False))
    val_x = np.array([x.numpy() for x, _ in val_set])
    val_y = np.array([y.numpy() for _, y in val_set])
    val_lwlrap_score = lwlrap_metric(val_y, model.predict(val_x)).numpy()
    print(f'Time to Predict Validation: {round(time.perf_counter() - tic, 2)} seconds')
    print(f'Validation Score: {val_lwlrap_score}')
    del val_set, val_x, val_y

    # Test Data Inference
    tic = time.perf_counter()
    test_predict = np.zeros([len(x_test), 24])

    for count, idx in enumerate(test_idx):
        idx_start = idx[0]
        idx_end = idx[1]
        x_sample_test = x_test[idx_start:idx_end]
        y_sample_test = np.zeros([len(x_sample_test), 24, ])
        test_set = tf.data.Dataset.from_tensor_slices((x_sample_test, y_sample_test))
        test_set = test_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, seed=42,
                                                                                   training=False))
        test_x = np.array([x.numpy() for x, _ in test_set])
        test_predict[idx_start:idx_end] = model.predict(test_x)
        print(f'Test Prediction for Count {count} of {len(test_idx)}')
        del test_set, x_sample_test, y_sample_test, test_x
    print(f'Time to Predict Test: {round(time.perf_counter() - tic, 2)} seconds')

    return test_predict, val_lwlrap_score


# Reshape test prediction for submission
def reshape_test_prediction(y):
    submission_matrix = np.zeros([num_test_samples, 24])
    N = num_windows * num_test_samples
    if N == len(y):
        print(f'Test Prediction Matrix Length is Correct {N}')
    else:
        print('An Error is Occuring')

    for i in range(num_test_samples):
        start_idx = i * num_windows
        end_idx = (i * num_windows) + num_windows
        submission_matrix[i] = y[start_idx:end_idx].max(axis=0)

    return submission_matrix


# Create submission file
def create_submission(y, file_names):
    sample_sub_path = r'C:\Kaggle\RainForest_R0\rfcx-species-audio-detection\sample_submission.csv'
    sample_sub = pd.read_csv(sample_sub_path)
    file_names = [file_name[:-5] for file_name in file_names]
    recording_ids = sample_sub['recording_id'].tolist()

    if file_names == recording_ids:
        submission = sample_sub.copy()
        submission.iloc[:, 1:] = y
    print('Submission File was Created')

    return submission


# Index Extraction for Test Data Predictions (performed b/c of memory limitation)
def index_extraction_test_data(N, interval_size):

    A = np.arange(0, N, interval_size)
    B = []
    for i, _ in enumerate(A):
        if i == len(A) - 1:
            B.append([A[i], N])
        else:
            B.append([A[i], A[i + 1]])

    return B


"""" User Inputs """
DATA_FOLDER = r'C:\Kaggle\RainForest_R0\HPTuning\224_512\ResNet18\ResNet18_7'
MODEL = 'ResNet18_7'
CHECK_TEST_IMAGES = True
INTERVAL_SIZE = 500

""" Load and Preprocess Test Data """
LWLRAP(24)
train, test = load_datasets(DATA_FOLDER)
num_windows = test.X.shape[1]
num_test_samples = test.X.shape[0]
test_file_names = test.file_names
test_X = np.zeros([num_windows * num_test_samples, test.X.shape[2], test.X.shape[3]])
for num in range(num_test_samples):
    sample_idx_start = num * num_windows
    sample_idx_end = (num * num_windows) + num_windows
    test_X[sample_idx_start:sample_idx_end, :, :] = test.X[num]
del test


""" Index Values for Iterating Test Data Set """
test_idxs = index_extraction_test_data(test_X.shape[0], INTERVAL_SIZE)


""" List of Models to Use for Inference """
list_of_models = list_of_models(DATA_FOLDER, MODEL)

""" Start Inference Loop and Predict for Each Fold/Validation Model """
val_scores = []
test_prediction = np.zeros([test_X.shape[0], 24])
for i, model_name in enumerate(list_of_models):
    print(f'Inference for Model: {model_name}')
    output_bias, val_idx = load_pickle(DATA_FOLDER, MODEL, model_name)
    model = load_tf_model(DATA_FOLDER, MODEL, model_name, output_bias)

    # Inferences on Test Dataset
    # TODO add validation to inference (put idx in folder with model as pickle)
    test_predict_single_model, val_score = inference(model, test_X, test_idxs, train.X[val_idx], train.Y[val_idx])
    test_prediction += test_predict_single_model
    val_scores.append(val_score)
    del model

test_prediction /= len(list_of_models)

""" Print Validation Scores for All Models """
for i in range(len(list_of_models)):
    print(f'{list_of_models[i]}: {val_scores[i]}')
print(f'Avg. Validation Score: {sum(val_scores) / len(val_scores)}')

""" Generate Submission/Test Prediction """
submission_prediction = reshape_test_prediction(test_prediction)

submission_df = create_submission(submission_prediction, test_file_names)
submission_df.to_csv(os.path.join(os.path.join(DATA_FOLDER, MODEL), 'submission.csv'), index=False)
print('Complete')
print('End of Script')
