import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import random
from classification_models.tfkeras import Classifiers


# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Preprocess TF datasets for training or validation
def preprocess(image, label, seed, inputs, training=False):

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Resize Images
    if inputs.model == 'ResNet34':
        height, width = 224, 224
    if inputs.model == 'ResNet50':
        height, width = 224, 224
    elif inputs.model == 'ResNet18':
        height, width = 224, 224
    elif inputs.model == 'EfficientNetB0':
        height, width = 224, 224
    elif inputs.model == 'EfficientNetB3':
        height, width = 300, 300
    elif inputs.model == 'MobileNetV2':
        height, width = 224, 224
    elif inputs.model == 'MobileNetV3Large':
        height, width = 224, 224
    elif inputs.model == 'DenseNet121':
        height, width = 224, 224
    elif inputs.model == 'Xception':
        height, width = 299, 299
    elif inputs.model == 'VGG16':
        height, width = 224, 224
    elif inputs.model == 'VGG19':
        height, width = 224, 224

    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [height, width])
    image = (image - tf.reduce_min(image)) / (
                tf.reduce_max(image) - tf.reduce_min(image)) * 255.0  # rescale to [0, 255]
    image = tf.image.grayscale_to_rgb(image)

    @tf.function
    def _specaugment(image, ERASE_TIME, ERASE_MEL):
        image = tf.expand_dims(image, axis=0)
        xoff = tf.random.uniform([2], minval=ERASE_TIME // 2, maxval=width - ERASE_TIME // 2, dtype=tf.int32)
        xsize = tf.random.uniform([2], minval=ERASE_TIME // 2, maxval=ERASE_TIME, dtype=tf.int32)
        yoff = tf.random.uniform([2], minval=ERASE_MEL // 2, maxval=height - ERASE_MEL // 2, dtype=tf.int32)
        ysize = tf.random.uniform([2], minval=ERASE_MEL // 2, maxval=ERASE_MEL, dtype=tf.int32)
        image = tfa.image.cutout(image, [height, xsize[0]], offset=[height // 2, xoff[0]])
        image = tfa.image.cutout(image, [height, xsize[1]], offset=[height // 2, xoff[1]])
        image = tfa.image.cutout(image, [ysize[0], width], offset=[yoff[0], width // 2])
        image = tfa.image.cutout(image, [ysize[1], width], offset=[yoff[1], width // 2])
        image = tf.squeeze(image, axis=0)
        return image

    if training:
        print('Training: Preprocessing Images')
        # gaussian
        if inputs.training['Gaussian'] is not None:
            gau = tf.keras.layers.GaussianNoise(inputs.training['Gaussian'])
            image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image, training=True), lambda: image)
        # brightness
        if inputs.training['Brightness'] is not None:
            image = tf.image.random_brightness(image, inputs.training['Brightness'])
        # specaugment
        if inputs.training['SpecAug'][0] is not None:
            erase_time = inputs.training['SpecAug'][0]
            erase_mel = inputs.training['SpecAug'][1]
            image = tf.cond(tf.random.uniform([]) < 0.5, lambda: _specaugment(image, erase_time, erase_mel),
                            lambda: image)

    # Select preprocess input function
    if inputs.model == 'ResNet34':
        _, preprocess_input = Classifiers.get('resnet34')
    elif inputs.model == 'ResNet18':
        _, preprocess_input = Classifiers.get('resnet18')
    elif inputs.model == 'ResNet50':
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label
    elif inputs.model == 'EfficientNetB0' or inputs.model == 'EfficientNetB3':
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label
    elif inputs.model == 'MobileNetV2':
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label
    elif inputs.model == 'MobileNetV3Large':
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        return image, label
    elif inputs.model == 'Xception':
        image = tf.keras.applications.xception.preprocess_input(image)
        return image, label
    elif inputs.model == 'DenseNet121':
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label
    elif inputs.model == 'VGG16':
        image = tf.keras.applications.vgg16.preprocess_input(image)
        return image, label
    elif inputs.model == 'VGG19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return image, label
    else:
        print('Model Type Not Found')
    image = preprocess_input(image)
    return image, label


def preprocess_mixup(ds):
    @tf.function
    def _mixup(inp, targ):
        indice = tf.range(len(inp))
        indice = tf.random.shuffle(indice)
        sinp = tf.gather(inp, indice, axis=0)
        starg = tf.gather(targ, indice, axis=0)
        alpha = 0.2
        t = tf.compat.v1.distributions.Beta(alpha, alpha).sample([len(inp)])
        tx = tf.reshape(t, [-1, 1, 1, 1])
        ty = tf.reshape(t, [-1, 1])
        x = inp * tx + sinp * (1 - tx)
        y = targ * ty + starg * (1 - ty)
        return x, y


# Plot preprocessed images to check preprocessing
def check_image(dataset, xtrain):
    plt.ion()
    plt.figure(1)

    count = 0
    for ele in dataset:
        species_id = ele[1].numpy().argmax()
        image_ds = ele[0].numpy()[:, :, 2]

        image_train = xtrain[count]
        max_image_ds = str(np.round(image_ds.max(), 1))[0:5]
        min_image_ds = str(np.round(image_ds.min(), 1))[0:5]

        max_image = str(np.round(image_train.max(), 1))[0:5]
        min_image = str(np.round(image_train.min(), 1))[0:5]

        plt.clf()
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        pos0 = ax0.imshow(image_train, vmin=0.0, vmax=255.0)
        pos1 = ax1.imshow(image_ds, vmin=-105.0, vmax=150.0)

        ax0.set_title(f'Numpy: {species_id} - Min {min_image} and Max {max_image}')
        ax1.set_title(f'Dataset: {species_id} - Min {min_image_ds} and Max {max_image_ds}')
        plt.colorbar(pos0, ax=ax0)
        plt.colorbar(pos1, ax=ax1)
        plt.draw()
        plt.waitforbuttonpress()
        count += 1


# Plot preprocessed images to check preprocessing
def check_test_image(dataset, xtrain):
    plt.ion()
    plt.figure(1)

    count = 0
    for ele in dataset:
        images_ds = ele[0].numpy()[:, :, :, 2]
        images_train = xtrain[count]
        samples_per_audio_file = images_train.shape[0]
        for i in range(samples_per_audio_file):
            image_ds = images_ds[i]
            image_train = images_train[i]
            max_image_ds = str(np.round(image_ds.max(), 1))[0:5]
            min_image_ds = str(np.round(image_ds.min(), 1))[0:5]

            max_image = str(np.round(image_train.max(), 1))[0:5]
            min_image = str(np.round(image_train.min(), 1))[0:5]

            plt.clf()
            ax0 = plt.subplot(121)
            ax1 = plt.subplot(122)

            pos0 = ax0.imshow(image_train, vmin=0.0, vmax=255.0)
            pos1 = ax1.imshow(image_ds, vmin=-105.0, vmax=150.0)

            ax0.set_title(f'Numpy Sample {i}: Min {min_image} and Max {max_image}')
            ax1.set_title(f'Dataset: Sample {i}: Min {min_image_ds} and Max {max_image_ds}')
            plt.colorbar(pos0, ax=ax0)
            plt.colorbar(pos1, ax=ax1)
            plt.draw()
            plt.waitforbuttonpress()
            count += 1


# Plot preprocessed images to check preprocessing
def check_test_image2(dataset, xtrain, num_windows):
    plt.ion()
    plt.figure(1)

    count = 0
    for ele in dataset:
        image_ds = ele[0].numpy()[:, :, 0]
        image_train = xtrain[count]

        max_image_ds = str(np.round(image_ds.max(), 1))[0:5]
        min_image_ds = str(np.round(image_ds.min(), 1))[0:5]

        max_image = str(np.round(image_train.max(), 1))[0:5]
        min_image = str(np.round(image_train.min(), 1))[0:5]

        plt.clf()
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        pos0 = ax0.imshow(image_train, vmin=0.0, vmax=255.0)
        pos1 = ax1.imshow(image_ds, vmin=-105.0, vmax=150.0)

        ax0.set_title(f'Numpy Sample {count}: Min {min_image} and Max {max_image}')
        ax1.set_title(f'Dataset: Sample {count}: Min {min_image_ds} and Max {max_image_ds}')
        plt.colorbar(pos0, ax=ax0)
        plt.colorbar(pos1, ax=ax1)
        plt.draw()
        plt.waitforbuttonpress()
        count += 1