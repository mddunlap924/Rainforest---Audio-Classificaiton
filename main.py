import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from lwlrap import LWLRAP
from Datasets.Mel_224_512.ResNeST50_R1 import model_setup
from Datasets.Mel_224_512.ResNeST50_R1 import preprocess_dataset


# Plot Metrics
def plot_few_metrics(history, save_path, lwlrap_score):
    metrics = ['loss', 'bc_loss', 'lwlrap', 'f1']
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'LWLRAP: {lwlrap_score}')
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(1, 4, n + 1)
        plt.plot(history.epoch, history.history[metric], color='r', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color='b', linestyle="--", label='Val')
        if metric == 'lwlrap':
            plt.axhline(lwlrap_score, color='black')
            plt.axhline(0.85, color='green')
            plt.title(f"{max(history.history['val_' + metric])}")
        if metric == 'loss':
            plt.axhline(min(history.history['val_' + metric]), color='black')
            plt.title(f"{min(history.history['val_' + metric])}")
        if metric == 'bc_loss':
            plt.axhline(min(history.history['val_' + metric]), color='black')
            plt.title(f"{min(history.history['val_' + metric])}")
        if metric == 'f1':
            plt.axhline(max(history.history['val_' + metric]), color='black')
            plt.title(f"{max(history.history['val_' + metric])}")
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
    plt.savefig(save_path[:-5] + '.png')


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


# TF model file path (file name) to save as during training
def create_model_save_name(data_folder, model_name, fold_num, folds, seed):

    file_directory = os.path.join(data_folder, model_name)

    fold_num = str(fold_num + 1)
    folds = str(folds)
    seed = str(seed)
    file_name = 'Fold' + fold_num + '_' + folds + '_R' + seed

    # Create folder to save checkpoint data
    os.mkdir(os.path.join(file_directory, file_name))

    file_path = os.path.join(os.path.join(file_directory, file_name), file_name + '.ckpt')

    return file_path


# Metric Score for Batched Dataset
def score_lwlrap_batch(x, y, ds, model):
    lwlrap_metric = LWLRAP(24)
    ds_unbatch = ds.unbatch()
    features_true = np.array([x.numpy() for x, _ in ds_unbatch])
    labels_true = np.array([y.numpy() for _, y in ds_unbatch])
    lwlrap_score = lwlrap_metric(labels_true, model.predict(features_true))
    return lwlrap_score


# Show output from model layers for analysis
def show_model_layers(x, y, model):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED, training=False))
    features_true = np.array([x.numpy() for x, _ in ds])

    data_in = np.expand_dims(features_true[0], 0)
    layer_nums = range(len(model.layers) - 20, len(model.layers), 1)
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


# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
""" User Inputs """
SEED = 42
DATA_FOLDER = r'C:\Kaggle\RainForest_R0\Datasets\Mel_224_512'
MODEL = 'ResNeST50_R1'
CHECK_TRAIN_IMAGES = False

""" Get other inputs that are local to a particular model """
model_inputs = model_setup.model_inputs()
FOLDS = model_inputs['folds']
BATCH_SIZE = model_inputs['batch_size']
REMOVE_MULTILABELS = model_inputs['remove_multilabels']

""" Seed everything """
seed_everything(SEED)

""" Load data """
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
# Calculate output bias
pos_rate = np.log(Y.mean(axis=0))
output_bias = tf.keras.initializers.Constant(pos_rate)
output_bias = tf.keras.initializers.Constant(-4.5)

""" Training Loop for Each Fold"""
histories = []
lwlrap_vals = []
for FOLD_IDX in range(FOLDS):
    x_train = X[train_idx[FOLD_IDX], :, :]
    y_train = Y[train_idx[FOLD_IDX], :]
    x_val = X[test_idx[FOLD_IDX], :, :]
    y_val = Y[test_idx[FOLD_IDX], :]

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED, training=True))
    if CHECK_TRAIN_IMAGES:
        preprocess_dataset.check_image(train_set, x_train)
    train_set = train_set.batch(BATCH_SIZE)
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_set = val_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED, training=False))
    val_set = val_set.batch(BATCH_SIZE)

    # Get the model
    model = model_setup.resnest50(output_bias)

    # Callbacks for model
    x_shape = [x_train.shape[0], x_train.shape[1]]
    monitor = {'early_stop': 'val_loss',
               'check_point': 'val_f1'}
    save_h5_path = create_model_save_name(DATA_FOLDER, MODEL, FOLD_IDX, FOLDS, SEED)
    callbacks = model_setup.model_callbacks(x_shape, BATCH_SIZE, monitor, save_h5_path)

    print(f'{MODEL}: Fold {FOLD_IDX + 1} of {FOLDS}')
    history = model.fit(train_set,
                        validation_data=val_set,
                        epochs=80,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks,
                        verbose=1)
    lwlrap_val = score_lwlrap_batch(x_val, y_val, val_set, model)
    print(f'{save_h5_path}  Val. Score: {lwlrap_val} \n')

    plot_few_metrics(history, save_h5_path, lwlrap_val)
    histories.append(history.history)
    lwlrap_vals.append([save_h5_path, lwlrap_val])

    """ Print Validation Losses """
    for ele in enumerate(lwlrap_vals):
        print(f'{ele[1][0]}: {ele[1][1]}')

    # Save variables to fold folder
    save_variables = {'train_idx': train_idx[FOLD_IDX],
                      'val_idx': test_idx[FOLD_IDX],
                      'folds': FOLDS,
                      'output_bias': output_bias}
    fold_folder = save_h5_path[:-5].split("\\")[-1]
    save_pickle(DATA_FOLDER, MODEL, fold_folder, 'save_variables.pickle', save_variables)

    del model, history

""" Print Validation Losses """
lwlrap_vals_scores = []
for ele in enumerate(lwlrap_vals):
    lwlrap_vals_scores.append(ele[1][1].numpy())
    print(f'{ele[1][0]}: {ele[1][1]}')
print(f'Avg. Val LWLRAP: {sum(lwlrap_vals_scores) / len(lwlrap_vals_scores)}')

print('End to Training')
