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
import HPTuning.HP_Info as HP_Info


# Plot Metrics
def plot_few_metrics(history, save_path, model_name, model_result):
    metrics = ['loss', 'bc_loss', 'lwlrap', 'f1']
    fig = plt.figure(figsize=(20, 10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(1, 4, n + 1)
        plt.plot(history.epoch, history.history[metric], color='r', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color='b', linestyle="--", label='Val')
        plt.axhline(model_result[metric], color='black')
        if metric is 'lwlrap':
            plt.axhline(0.85, color='green')
        plt.title(f'{model_result[metric]}')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, 0.4])
        else:
            plt.ylim([0, 1])
        plt.legend()
    plt.savefig(os.path.join(save_path, model_name + '.png'))


# Save python variables to cwd with given filename
def save_pickle(data_folder, model_name, fold_folder, file_name, variables):
    save_path = os.path.join(os.path.join(data_folder, model_name), fold_folder)

    with open(os.path.join(save_path, file_name), 'wb') as output:
        pickle.dump(variables, output, pickle.HIGHEST_PROTOCOL)


# Load datafiles in this directory
def load_data(data_folder):
    files_in_directory = os.listdir(data_folder)
    for file in files_in_directory:
        if 'training' in file:
            file_name = file

    train_data_path = os.path.join(data_folder, file_name)
    with open(train_data_path, 'rb') as input_file:
        train = pickle.load(input_file)

    return train


# Show output from model layers for analysis
def show_model_layers(x, y, model, model_inputs):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, 42, model_inputs, training=False))
    features_true = np.array([x.numpy() for x, _ in ds])

    data_in = np.expand_dims(features_true[1], 0)
    layer_nums = range(len(model.layers) - 20, len(model.layers), 1)
    # layer_nums = range(len(model.layers) - 7, len(model.layers), 1)
    layer_outputs = {}
    for layer_num in layer_nums:
        intermediate_layer_model_output = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
            index=layer_num).output)
        intermediate_layer_model_input = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
            index=layer_num).input)
        intermediate_output = intermediate_layer_model_output.predict(data_in)
        intermediate_input = intermediate_layer_model_input.predict(data_in)
        layer_name = model.get_layer(index=layer_num).name
        layer_outputs[layer_num] = {'name': layer_name,
                                    'input': intermediate_input,
                                    'output': intermediate_output}

    return layer_outputs


# TF model file path (file name) to save as during training
def create_model_save_name(input, fold_num):
    model_name = input.save_path.split('\\')[-1] + '__Fold' + str(fold_num + 1) + '_' + str(input.folds)
    save_path = os.path.join(input.save_path, model_name)

    # Create folder to save checkpoint data
    # TODO could mess up here is upper directories exist
    if not os.path.exists(input.save_path):
        os.makedirs(input.save_path)

    file_path = os.path.join(save_path, model_name + '.ckpt')

    return file_path, model_name


# Load Tensorflow model with weights
def load_tf_model(model_path, input):
    # Select model
    model = model_setup.resnet_model(input)

    # Load weights into the model
    model.load_weights(model_path)

    return model


# Metric Score for Batched Dataset
def score_model(x, y, model_path, seed, model_input):
    # lwlrap_metric = LWLRAP(24)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, seed, model_input, training=False))
    x_ds = np.array([X.numpy() for X, _ in ds])
    y_ds = np.array([Y.numpy() for _, Y in ds])
    model = load_tf_model(model_path, model_input)

    model.evaluate(x_ds, y_ds, batch_size=1)

    results_eval = model.evaluate(x_ds, y_ds, batch_size=1)

    model_eval_results = {}
    for i, metric in enumerate(model.metrics):
        model_eval_results[metric.name] = results_eval[i]

    model_eval_results['path'] = model_path
    model_eval_results['y_predict'] = model.predict(x_ds)

    del model
    tf.keras.backend.clear_session()

    return model_eval_results


# Mixup Augmentation for Training Data
def mixup_augmentation(x, y, alpha, seed):

    np.random.seed(seed)
    arr = np.arange(len(x))
    np.random.shuffle(arr)
    if not len(arr) % 2:
        no_mixup_idx = arr[0:len(arr) // 2]
        mixup_idx = arr[len(arr) // 2:]
    else:
        no_mixup_idx = arr[0:len(arr) // 2]
        mixup_idx = arr[(len(arr) // 2) + 1:]

    x_no_mixup = x[no_mixup_idx]
    y_no_mixup = y[no_mixup_idx]

    x_mixup = x[mixup_idx]
    y_mixup = y[mixup_idx]

    lams = np.random.beta(alpha, alpha, size=len(x_mixup))
    ori_index = np.arange(int(len(x_mixup)))
    index_array = np.arange(int(len(x_mixup)))
    np.random.shuffle(index_array)
    x_mix = np.empty(x_mixup.shape, dtype=np.float32)
    y_mix = np.empty(y_mixup.shape, dtype=np.float32)

    for i, lam in enumerate(lams):
        x_mix[i] = lam * x_mixup[ori_index[i]] + (1 - lam) * x_mixup[index_array[i]]
        y_mix[i] = lam * y_mixup[ori_index[i]] + (1 - lam) * y_mixup[index_array[i]]

    x_ = np.concatenate((x_no_mixup, x_mix))
    y_ = np.concatenate((y_no_mixup, y_mix))
    shuffle_idx = np.arange(len(x_))
    np.random.shuffle(shuffle_idx)
    x_ = x_[shuffle_idx]
    y_ = y_[shuffle_idx]

    # #TODO add variation here so it can be 1/2mix and 1/2original, 100%mixup, etc.
    # x_, y_ = x_mix, y_mix

    return x_, y_

# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(inputs, SEED):
    REMOVE_MULTILABELS = False
    CHECK_TRAIN_IMAGES = False
    FOLDS = inputs.folds
    output_bias = inputs.output_bias
    BATCH_SIZE = inputs.batch_size

    """ Seed everything """
    seed_everything(SEED)

    """ Load Dataset for Training """
    if inputs.dataset == '224_512':
        DATA_FOLDER = r'C:\Kaggle\RainForest_R0\Datasets\Mel_224_512'
    if inputs.dataset == '224_768':
        DATA_FOLDER = r'C:\Kaggle\RainForest_R0\Datasets\Mel_224_768'

    else:
        DATA_FOLDER = os.path.join(r'C:\Kaggle\RainForest_R0\Datasets', 'Mel_' + inputs.dataset)
        print(f'Data Folder: {DATA_FOLDER}')

    train = load_data(DATA_FOLDER)
    X = train.X
    Y = train.Y
    if REMOVE_MULTILABELS:
        idx_remove = (Y.sum(axis=1) > 1).nonzero()[0]
        X = np.delete(X, idx_remove, axis=0)
        Y = np.delete(Y, idx_remove, axis=0)

    """ Stratify data for training """
    # Y labels for stratified k-fold
    y_labels = np.argmax(Y, axis=1)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    train_idx = []
    test_idx = []
    for train_index, test_index in skf.split(X, y_labels):
        train_idx.append(train_index)
        test_idx.append(test_index)

    """ Prepare for training"""
    # # Calculate output bias
    # pos_rate = np.log(Y.mean(axis=0))
    # output_bias = tf.keras.initializers.Constant(pos_rate)

    """ Training Loop for Each Fold"""
    histories = []
    results = []

    for FOLD_IDX in range(FOLDS):
        x_train = X[train_idx[FOLD_IDX], :, :]
        y_train = Y[train_idx[FOLD_IDX], :]
        x_val = X[test_idx[FOLD_IDX], :, :]
        y_val = Y[test_idx[FOLD_IDX], :]

        if inputs.mixup is not None:
            x_train, y_train = mixup_augmentation(x_train, y_train, inputs.mixup, SEED)

        train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_set = train_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED,
                                                                                     inputs, training=True))
        if CHECK_TRAIN_IMAGES:
            preprocess_dataset.check_image(train_set, x_train)
        train_set = train_set.batch(BATCH_SIZE)
        val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_set = val_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED,
                                                                                 inputs, training=False))
        val_set = val_set.batch(BATCH_SIZE)

        # Get the model
        # model = model_setup.resnet_model(inputs)
        model = model_setup.sed_model2(inputs)

        # Show model layers prior to training
        # model_layers = show_model_layers(x_val, y_val, model, inputs)

        # Callbacks for model
        x_shape = [x_train.shape[0], x_train.shape[1]]
        monitor = {'early_stop': 'val_loss',
                   'check_point': 'val_f1'}
        save_h5_path, model_name = create_model_save_name(inputs, FOLD_IDX)
        callbacks = model_setup.model_callbacks(x_shape, BATCH_SIZE, monitor, save_h5_path, inputs)

        print(f'{model_name}: Fold {FOLD_IDX + 1} of {FOLDS}')
        history = model.fit(train_set,
                            validation_data=val_set,
                            epochs=50,
                            batch_size=BATCH_SIZE,
                            callbacks=callbacks,
                            verbose=1)

        # Show model layers after training
        model_layers = show_model_layers(x_val, y_val, model, inputs)

        del model

        # Evaluate Model from Checkpoint Weights
        result = score_model(x_val, y_val, save_h5_path, SEED, inputs)

        # Append Variables for later saving
        histories.append(history.history)
        results.append(result)

        # Save an Image of the History
        plot_few_metrics(history, inputs.save_path, model_name, result)

        # Print Outputs for Viewing
        for ele in results:
            print(f'{ele["path"]} - LWLRAP: {ele["lwlrap"]}')

        tf.keras.backend.clear_session()

    # Save variables to fold folder
    save_variables = {'train_idx': train_idx,
                      'val_idx': test_idx,
                      'histories': histories,
                      'results': results,
                      'y_true': Y
                      }

    return save_variables

print('End to Training')
