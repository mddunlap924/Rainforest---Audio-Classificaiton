import pickle
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from HPTuning import model_setup_sed as model_setup
from HPTuning import preprocess_dataset_sed as preprocess_dataset
import functools
import tensorflow_probability as tfp
import gc
from misc_methods import plot_sed_few_metrics, plot_no_sed_few_metrics
import math
from dataset3 import Dataset


# Save python variables to cwd with given filename
def save_pickle(data_folder, model_name, fold_folder, file_name, variables):
    save_path = os.path.join(os.path.join(data_folder, model_name), fold_folder)

    with open(os.path.join(save_path, file_name), 'wb') as output:
        pickle.dump(variables, output, pickle.HIGHEST_PROTOCOL)


# Load datafiles in this directory
def load_data(data_folder, *, augment=False, fp=False):

    if augment:
        string_check = 'training_augment.p'
    else:
        string_check = 'training.p'
        if fp:
            string_check = 'training_fp.p'

    files_in_directory = os.listdir(data_folder)
    for file in files_in_directory:
        if string_check in file:
            file_name = file

    train_data_path = os.path.join(data_folder, file_name)
    with open(train_data_path, 'rb') as input_file:
        train = pickle.load(input_file)

    return train


# # Show output from model layers for analysis
# def show_model_layers(x, y, model, model_inputs):
#     ds = tf.data.Dataset.from_tensor_slices((x, y))
#     ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, 42, model_inputs, training=False))
#     features_true = np.array([x.numpy() for x, _ in ds])
#
#     data_in = np.expand_dims(features_true[1], 0)
#     layer_nums = range(len(model.layers) - 20, len(model.layers), 1)
#     # layer_nums = range(len(model.layers) - 7, len(model.layers), 1)
#     layer_outputs = {}
#     for layer_num in layer_nums:
#         intermediate_layer_model_output = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
#             index=layer_num).output)
#         intermediate_layer_model_input = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
#             index=layer_num).input)
#         intermediate_output = intermediate_layer_model_output.predict(data_in)
#         intermediate_input = intermediate_layer_model_input.predict(data_in)
#         layer_name = model.get_layer(index=layer_num).name
#         layer_outputs[layer_num] = {'name': layer_name,
#                                     'input': intermediate_input,
#                                     'output': intermediate_output}
#
#
#     return layer_outputs


# Show output from model layers for analysis
def show_model_layers(x, y, model, model_inputs):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, 42, model_inputs, training=False))
    features_true = np.array([x.numpy() for x, _ in ds])

    data_in = np.expand_dims(features_true[1], 0)
    layer_outputs = {}
    layer_names = ['end_norm_att', 'frame', 'clip', 'segmentwise_output']
    for i, layer_num in enumerate(layer_names):
        intermediate_layer_model_output = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
            name=layer_num).output)
        intermediate_layer_model_input = tf.keras.Model(inputs=model.input, outputs=model.get_layer(
            name=layer_num).input)
        intermediate_output = intermediate_layer_model_output.predict(data_in)
        intermediate_input = intermediate_layer_model_input.predict(data_in)
        layer_name = model.get_layer(name=layer_num).name
        layer_outputs[i] = {'name': layer_name,
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
    if input.sed:
        model = model_setup.sed_model(input)
    else:
        model = model_setup.no_sed_model(input)
        print('No SED Model was Loaded')

    # Load weights into the model
    model.load_weights(model_path)

    return model


# Metric Score for Batched Dataset
def score_model(x, y, model_path, seed, model_input):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda image, label: preprocess_dataset.preprocess(image, label, seed, model_input, training=False))
    x_ds = np.array([X.numpy() for X, _ in ds])
    y_ds = np.array([Y.numpy() for _, Y in ds])
    model = load_tf_model(model_path, model_input)

    model.evaluate(x_ds, y_ds, batch_size=1)

    results_eval = model.evaluate(x_ds, y_ds, batch_size=1)

    model_eval_results = {}
    for i, metric in enumerate(model.metrics):
        metric_name_org = metric.name
        if 'clip_clip' in metric_name_org or 'frame_frame' in metric_name_org:
            metric_name_split = metric_name_org.split('_')
            metric_name_new = metric_name_split[0] + '_' + metric_name_split[-1]
        else:
            metric_name_new = metric_name_org

        model_eval_results[metric_name_new] = results_eval[i]

    model_eval_results['path'] = model_path
    model_pred = model.predict(x_ds)
    # if isinstance(model_pred, list):
    #     model_eval_results['y_predict'] = {'clip': model_pred[0]}
    # if model_input.sed:
    #     model_eval_results['y_predict'] = {'frame': model_pred[0],
    #                                        'clip': model_pred[1]}

    if model_input.sed:
        model_eval_results['y_predict'] = {'frame': model_pred[0],
                                           'clip': model_pred[1]}
    else:
        model_eval_results['y_predict'] = {'frame': model_pred[0]}

    del model
    tf.keras.backend.clear_session()

    return model_eval_results


# Image Mixup by Batch
def mixup_no_sed(batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.
    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412
    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      images: A batch of images of shape [batch_size, ...]
      labels: A batch of labels of shape [batch_size, num_classes]
    Returns:
      A tuple of (images, labels) with the same dimensions as the input with
      Mixup regularization applied.
    """
    NUM_CLASSES = 24

    if tf.random.uniform([]) < 0.5:
        # return images, (tf.zeros([batch_size, 1, NUM_CLASSES]), labels)
        return images, labels
    mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
    # Mixup on a single batch is implemented by taking a weighted sum with the same batch in reverse.
    images_mix = (images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
    # labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    # # return images_mix, labels_mix #images, labels
    # return images_mix, (tf.zeros([batch_size, 1, NUM_CLASSES]), labels_mix) #images, labels

    labels_mix = tf.squeeze(labels * mix_weight + labels[::-1] * (1. - mix_weight))
    # labels_mix = (labels * mix_weight + labels[::-1] * (1. - mix_weight))
    return images_mix, labels_mix #images, labels


# Image Mixup by Batch
def mixup(batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.
    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412
    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      images: A batch of images of shape [batch_size, ...]
      labels: A batch of labels of shape [batch_size, num_classes]
    Returns:
      A tuple of (images, labels) with the same dimensions as the input with
      Mixup regularization applied.
    """
    NUM_CLASSES = 24

    if tf.random.uniform([]) < 0.5:
        # return images, (tf.zeros([batch_size, 1, NUM_CLASSES]), labels)
        return images, labels
    # labels = tf.expand_dims(labels, axis=-1)
    # print(f'lables {labels}')
    mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
    labels_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1])

    # Mixup on a single batch is implemented by taking a weighted sum with the same batch in reverse.
    images_mix = (images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
    labels_mix = (labels * labels_mix_weight + labels[::-1] * (1. - labels_mix_weight))
    return images_mix, labels_mix  # images, labels


# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(inputs, SEED):
    # REMOVE_MULTILABELS = False
    CHECK_TRAIN_IMAGES = False
    FOLDS = inputs.folds
    BATCH_SIZE = inputs.batch_size

    """ Seed everything """
    seed_everything(SEED)

    """ Load Dataset for Training """
    DATA_FOLDER = os.path.join(r'C:\Kaggle\RainForest_R0\Datasets', 'Mel_' + inputs.dataset)
    print(f'Data Folder: {DATA_FOLDER}')

    train = load_data(DATA_FOLDER)
    X = train.X
    Y = train.Y
    # train_file_names = train.file_names
    # train_species = [str(i[0]) for i in train.species]

    if inputs.augment:
        train_augment = load_data(DATA_FOLDER, augment=True)
        X_aug = train_augment.X
        Y_aug = train_augment.Y

    if inputs.fp:
        train_fp = load_data(DATA_FOLDER, fp=True)
        X_fp = train_fp.X
        Y_fp = train_fp.Y


    """ Stratify data for training """
    # Y labels for stratified k-fold
    y_labels = np.max(Y, axis=1)
    y_labels = np.argmax(y_labels, axis=1)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    train_idx = []
    test_idx = []
    for train_index, test_index in skf.split(X, y_labels):
        train_idx.append(train_index)
        test_idx.append(test_index)

    """ Training Loop for Each Fold"""
    histories = []
    results = []

    for FOLD_IDX in range(FOLDS):
        x_train = np.float32(X[train_idx[FOLD_IDX]])
        y_train = np.float32(Y[train_idx[FOLD_IDX]])
        x_val = np.float32(X[test_idx[FOLD_IDX]])
        y_val = np.float32(Y[test_idx[FOLD_IDX]])
        # val_files_species = np.core.defchararray.add(np.array(train_file_names)[test_idx[FOLD_IDX]], \
        #                     np.array(train_species)[test_idx[FOLD_IDX]])
        if inputs.augment:
            # Remove validation data from augmented files
            x_train_aug = np.float32(X_aug[train_idx[FOLD_IDX]])
            y_train_aug = np.float32(Y_aug[train_idx[FOLD_IDX]])
            x_train = np.concatenate((x_train, x_train_aug), axis=0)
            y_train = np.concatenate((y_train, y_train_aug), axis=0)

            # # Mix between non-aug and aug
            # np.random.seed(SEED)
            # mix_idx = np.arange(x_train.shape[0])
            # np.random.shuffle(mix_idx)
            # idx_cut = int(len(mix_idx)/1.5)
            # x_train = np.concatenate((x_train[0:idx_cut], x_train_aug[idx_cut:]), axis=0)
            # y_train = np.concatenate((y_train[0:idx_cut], y_train_aug[idx_cut:]), axis=0)
            # x_train = x_train
            # y_train = y_train

        # Reduce y_train and y_val to (samples x classes) for no sed models
        if not inputs.sed:
            y_train = np.max(y_train, axis=-2)
            y_val = np.max(y_val, axis=-2)

        if inputs.augment:
            # Shuffle Data
            shuffle_idx = np.arange(x_train.shape[0])
            np.random.seed(SEED)
            np.random.shuffle(shuffle_idx)
            x_train = x_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

        train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_set = train_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED,
                                                                                     inputs, training=True),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        if CHECK_TRAIN_IMAGES:
            preprocess_dataset.check_image(train_set, x_train)
        train_set = train_set.batch(BATCH_SIZE)

        if inputs.mixup is not None:
            if inputs.sed:
                train_set = train_set.map(functools.partial(mixup, BATCH_SIZE, inputs.mixup))
            else:
                train_set = train_set.map(functools.partial(mixup_no_sed, BATCH_SIZE, inputs.mixup))

        val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_set = val_set.map(lambda image, label: preprocess_dataset.preprocess(image, label, SEED,
                                                                                 inputs, training=False))
        val_set = val_set.batch(BATCH_SIZE)

        # Get the model
        if inputs.sed:
            model = model_setup.sed_model(inputs)
        else:
            model = model_setup.no_sed_model(inputs)

        # Show model layers prior to training
        # model_layers = show_model_layers(x_val, y_val, model, inputs)

        # Callbacks for model
        x_shape = [x_train.shape[0], x_train.shape[1]]
        if inputs.sed:
            monitor = {'early_stop': 'val_loss',
                       'check_point': 'val_loss'}
        else:
            monitor = {'early_stop': 'val_loss',
                       'check_point': 'val_lwlrap'}
        save_h5_path, model_name = create_model_save_name(inputs, FOLD_IDX)
        callbacks = model_setup.model_callbacks(x_shape, BATCH_SIZE, monitor, save_h5_path, inputs)

        print(f'{model_name}: Fold {FOLD_IDX + 1} of {FOLDS}')
        # model.summary()
        # steps_per_epoch = int((x_shape[0] / BATCH_SIZE))
        steps_per_epoch = int(math.ceil(x_shape[0] / BATCH_SIZE))
        history = model.fit(train_set,
                            validation_data=val_set,
                            epochs=inputs.epochs,
                            batch_size=BATCH_SIZE,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            verbose=1)

        # # Show model layers after training
        # model_layers = show_model_layers(x_val, y_val, model, inputs)

        # Evaluate Model from Checkpoint Weights
        result = score_model(x_val, y_val, save_h5_path, SEED, inputs)

        # Append Variables for later saving
        histories.append(history.history)
        results.append(result)

        # Save an Image of the History
        if inputs.sed:
            plot_sed_few_metrics(history, inputs.save_path, model_name, result)
        else:
            plot_no_sed_few_metrics(history, inputs.save_path, model_name, result)

        # Print Outputs for Viewing
        for ele in results:
            if inputs.sed:
                print(f'{ele["path"]} - LWLRAP: {ele["frame_lwlrap"]}')
            else:
                print(f'{ele["path"]} - LWLRAP: {ele["lwlrap"]}')

        del model, history
        tf.keras.backend.clear_session()
        gc.collect()

    # Save variables to fold folder
    save_variables = {'train_idx': train_idx,
                      'val_idx': test_idx,
                      'histories': histories,
                      'results': results,
                      'y_true': Y
                      }

    return save_variables
