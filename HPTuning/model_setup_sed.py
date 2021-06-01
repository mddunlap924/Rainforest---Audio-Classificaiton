import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)
import tensorflow_addons as tfa
from lwlrap_clip import LWLRAP_CLIP
from lwlrap_frame import LWLRAP_FRAME
from lwlrap import LWLRAP
from clr_callback import CyclicLR
from sgd_callback import SGDRScheduler
from classification_models.tfkeras import Classifiers
from cosine_warm_up import WarmUpCosineDecayScheduler
import tensorflow_probability as tfp
import math
import efficientnet.tfkeras as eff
from tensorflow.keras.callbacks import Callback
import numpy as np

ResNet34, _ = Classifiers.get('resnet34')
ResNet18, _ = Classifiers.get('resnet18')


# BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def frame_dice_loss(y_true, y_pred, smooth=1e-6):
    # https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
    # y_pred = y_true * y_pred
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce = BCE(y_true, y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    dice_loss = 1 - (2 * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)
    Dice_BCE = bce + dice_loss

    return Dice_BCE


def custom_binary_loss(y_true, y_pred):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_true = tf.cast(tf.math.reduce_max(y_true, axis=-2), dtype=tf.float32)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    y_true_idx = tf.where(y_true > 0)

    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1
    term_1 = y_true * K.log(y_pred + K.epsilon())  # Cancels out when target is 0
    t_random = tf.random.uniform(shape=[24], minval=0.95, maxval=1.0, dtype=tf.float32, seed=10)
    t_random[y_true_idx] = 1.0
    term_1 = tf.random.uniform(shape=[24], minval=0.95, maxval=1.0, dtype=tf.float32, seed=10) * term_1
    return -K.mean(term_0 + term_1, axis=1)


# https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/blob/master/train_models.py
def symmetric_cross_entropy(y_true, y_pred):
    alpha = 6.0
    beta = 1.0
    y_true = tf.cast(tf.math.reduce_max(y_true, axis=-2), dtype=tf.float32)
    y_true_1 = y_true
    y_pred_1 = y_pred

    y_true_2 = y_true
    y_pred_2 = y_pred

    y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
    y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)
    loss = alpha * tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis=-1)) + \
           beta * tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis=-1))

    return loss


def symmetric_cross_entropy_frame(y_true, y_pred):
    alpha = 6.0
    beta = 1.0

    y_true_1 = y_true
    y_pred_1 = y_pred

    y_true_2 = y_true
    y_pred_2 = y_pred

    y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
    y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)
    loss = alpha * tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis=-1)) + \
           beta * tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis=-1))

    return loss


# Framewise Output - Custom loss function
def frame_loss_bce(y_true, y_pred):
    # y_pred_ = y_true * y_pred
    y_pred_ = y_pred
    y_true_ = y_true
    # y_true_ = tf.keras.backend.flatten(y_true_)
    # y_pred_ = tf.keras.backend.flatten(y_pred_)
    # y_pred_ = tf.math.reduce_max(y_pred_, axis=-2)
    # y_true_ = tf.math.reduce_max(y_true, axis=-2)

    # y_true_ = y_true
    # y_pred_ = y_pred

    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # BCE = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.75)
    # CCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    frame_loss = BCE(y_true_, y_pred_)
    return frame_loss


# Clipwise Output - Custom loss function
def clip_loss_bce(y_true, y_pred):
    y_true_ = tf.cast(tf.math.reduce_max(y_true, axis=-2), dtype=tf.float32)
    # BCE = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.95)
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    clip_loss = BCE(y_true_, y_pred)
    return clip_loss


# Masked Loss Funcation
from keras import backend as K


def masked_loss_function(y_true, y_pred):
    # mask_value = tf.constant(-1.0)
    # mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    # bce = K.binary_crossentropy(y_true * mask, y_pred * mask)
    bce = K.binary_crossentropy(y_true, y_pred * y_true)
    return bce


# SED Models
def sed_model(inputs):
    METRICS = [[LWLRAP_FRAME(24)], [LWLRAP_CLIP(24)]]
    interp_width = inputs.width
    data_format = 'channels_last'

    # Select model
    if inputs.model == 'ResNet34':
        height, width = 224, 224
        base_model = ResNet34(input_shape=(height, width, 3),
                              include_top=False,
                              weights='imagenet')
    elif inputs.model == 'ResNet18':
        height, width = 224, 224
        base_model = ResNet18(input_shape=(height, width, 3),
                              include_top=False,
                              weights='imagenet')
    elif inputs.model == 'ResNet50':
        height, width = 224, 224
        base_model = tf.keras.applications.ResNet50(input_shape=(height, width, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    elif inputs.model == 'EfficientNetB0':
        height, width = 224, 224
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(height, width, 3),
                                                          include_top=False,
                                                          weights='imagenet')
    elif inputs.model == 'EfficientNetB3':
        height, width = 300, 300
        base_model = tf.keras.applications.EfficientNetB3(input_shape=(height, width, 3),
                                                          include_top=False,
                                                          weights='imagenet')
    elif inputs.model == 'MobileNetV2':
        height, width = 224, 224
        base_model = tf.keras.applications.MobileNetV2(input_shape=(height, width, 3),
                                                       include_top=False,
                                                       weights='imagenet')
    elif inputs.model == 'MobileNetV3Large':
        height, width = 224, 224
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=(height, width, 3),
                                                            include_top=False,
                                                            weights='imagenet')
    elif inputs.model == 'DenseNet121':
        height, width = 224, 224
        base_model = tf.keras.applications.DenseNet121(input_shape=(height, width, 3),
                                                       include_top=False,
                                                       weights='imagenet')
    elif inputs.model == 'Xception':
        height, width = 299, 299
        base_model = tf.keras.applications.Xception(input_shape=(height, width, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    elif inputs.model == 'VGG16':
        height, width = 224, 224
        base_model = tf.keras.applications.VGG16(input_shape=(height, width, 3),
                                                 include_top=False,
                                                 weights='imagenet')
    elif inputs.model == 'VGG19':
        height, width = 224, 224
        base_model = tf.keras.applications.VGG16(input_shape=(height, width, 3),
                                                 include_top=False,
                                                 weights='imagenet')

    for layer in base_model.layers:
        if inputs.base_trainable:
            # print(f'Base Layers are Trainable')
            layer.trainable = inputs.base_trainable
        else:
            # print(f'Base Layers are NOT Trainable')
            layer.trainable = False

    # Aggregate over Frequency Axis
    x = tf.reduce_mean(base_model.output, axis=1)
    # (Batch Size, Time, Filters)

    # Max and Avg. Over the Filter Axis
    x1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x2 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = x1 + x2

    if inputs.fc_layer is not None:
        x = tf.keras.layers.Dropout(inputs.fc_layer['dropout0'])(x)
        x = tf.keras.layers.Dense(inputs.fc_layer['num_hidden'], activation='relu')(x)
        x = tf.keras.layers.Dropout(inputs.fc_layer['dropout1'])(x)

    # Classifier / Tagging / Segmentwise
    segmentwise_output = tf.keras.layers.Conv1D(filters=24,
                                                kernel_size=1,
                                                strides=1,
                                                padding='same',
                                                activation='sigmoid',
                                                name="segmentwise_output",
                                                )(x)

    # Attention Module / Class Probability with Softmax
    norm_att = tf.keras.layers.Conv1D(filters=24,
                                      kernel_size=1,
                                      strides=1,
                                      padding='same',
                                      name="start_norm_att",
                                      )(x)
    norm_att = tf.keras.activations.tanh(norm_att)
    norm_att = tf.keras.activations.softmax(norm_att, axis=-2)
    norm_att = tf.keras.layers.Lambda(lambda x: x, name='end_norm_att')(norm_att)

    # Clipwise output
    clip = tf.math.reduce_sum(norm_att * segmentwise_output, axis=1)
    clip = tf.keras.layers.Lambda(lambda x: x, name="clip")(clip)

    x_interp = tf.linspace(start=0.0, stop=interp_width - 1, num=interp_width)
    framewise = tfp.math.interp_regular_1d_grid(x=x_interp, x_ref_min=0.0, x_ref_max=interp_width,
                                                y_ref=segmentwise_output,
                                                axis=-2)
    framewise = tf.keras.layers.Lambda(lambda x: x, name="frame")(framewise)

    output = [framewise, clip]

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimization
    opt = tf.keras.optimizers.Adam()

    # Loss Function
    loss_fn = [frame_loss_bce, clip_loss_bce]
    # loss_fn = [symmetric_cross_entropy_frame, symmetric_cross_entropy]
    # loss_fn = [frame_loss_bce,  custom_binary_loss]

    # Compile the model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  loss_weights=inputs.loss_weights,
                  metrics=METRICS)
    # model.summary()
    return model


#
# # SED Models
# def sed_model(inputs):
#     METRICS = [[LWLRAP_FRAME(24)], [LWLRAP_CLIP(24)]]
#     interp_width = inputs.width
#     data_format = 'channels_last'
#
#     # Select model
#     if inputs.model == 'ResNet34':
#         height, width = 224, 512
#         base_model = ResNet34(input_shape=(height, width, 3),
#                               include_top=False,
#                               weights='imagenet')
#     elif inputs.model == 'ResNet18':
#         height, width = 224, 512
#         base_model = ResNet18(input_shape=(height, width, 3),
#                               include_top=False,
#                               weights='imagenet')
#     elif inputs.model == 'EfficientNetB0':
#         height, width = 224, 512
#         base_model = tf.keras.applications.EfficientNetB0(input_shape=(height, width, 3),
#                                                           include_top=False,
#                                                           weights='imagenet')
#     elif inputs.model == 'EfficientNetB3':
#         height, width = 300, 600
#         base_model = tf.keras.applications.EfficientNetB3(input_shape=(height, width, 3),
#                                                           include_top=False,
#                                                           weights='imagenet')
#     elif inputs.model == 'MobileNetV2':
#         height, width = 224, 512
#         base_model = tf.keras.applications.MobileNetV2(input_shape=(height, width, 3),
#                                                        include_top=False,
#                                                        weights='imagenet')
#     elif inputs.model == 'DenseNet121':
#         height, width = 224, 512
#         base_model = tf.keras.applications.DenseNet121(input_shape=(height, width, 3),
#                                                        include_top=False,
#                                                        weights='imagenet')
#
#     for layer in base_model.layers:
#         if inputs.base_trainable:
#             # print(f'Base Layers are Trainable')
#             layer.trainable = inputs.base_trainable
#         else:
#             # print(f'Base Layers are NOT Trainable')
#             layer.trainable = False
#
#     # Aggregate over Frequency Axis
#     x = tf.reduce_mean(base_model.output, axis=1)
#     # (Batch Size, Time, Filters)
#
#     # # Max and Avg. Over the Filter Axis
#     x1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
#     x2 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
#     x = x1 + x2
#     # x = tf.keras.layers.Dropout(0.5)(x)
#     # xa = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation='tanh', return_sequences=True))(x)
#     # xb = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation='sigmoid', return_sequences=True))(x)
#     # x = tf.keras.layers.concatenate([xa, xb])
#
#     if inputs.fc_layer is not None:
#         # x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Dropout(inputs.fc_layer['dropout0'])(x)
#         x = tf.keras.layers.Dense(inputs.fc_layer['num_hidden'], activation='relu')(x)
#         # x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Dropout(inputs.fc_layer['dropout1'])(x)
#
#     # Classifier / Tagging / Segmentwise
#     # segmentwise_output = tf.keras.layers.Dense(24, activation='sigmoid', name="segmentwise_output")(x)
#     segmentwise_output = tf.keras.layers.Dense(24, activation='sigmoid',
#                                                name="segmentwise_output",
#                                                bias_initializer=tf.keras.initializers.Constant(
#                                                    inputs.bias_initializer))(x)
#     # Attention Module / Class Probability with Softmax
#     norm_att = tf.keras.layers.Dense(24, name="start_norm_att",
#                                      bias_initializer=tf.keras.initializers.Constant(inputs.bias_initializer))(x)
#     # norm_att = tf.keras.layers.Dense(24, name="start_norm_att")(x)
#     norm_att = tf.keras.activations.tanh(norm_att)
#     norm_att = tf.keras.activations.softmax(norm_att, axis=-2)
#     norm_att = tf.keras.layers.Lambda(lambda x: x, name='end_norm_att')(norm_att)
#
#     # Clipwise output
#     clip = tf.math.reduce_sum(norm_att * segmentwise_output, axis=1)
#     clip = tf.keras.layers.Lambda(lambda x: x, name="clip")(clip)
#
#     x_interp = tf.linspace(start=0.0, stop=interp_width - 1, num=interp_width)
#     framewise = tfp.math.interp_regular_1d_grid(x=x_interp, x_ref_min=0.0, x_ref_max=interp_width,
#                                                 y_ref=segmentwise_output,
#                                                 axis=-2)
#     framewise = tf.keras.layers.Lambda(lambda x: x, name="frame")(framewise)
#
#     output = [framewise, clip]
#
#     model = tf.keras.Model(inputs=base_model.input, outputs=output)
#
#     # Optimization
#     opt = tf.keras.optimizers.Adam()
#
#
#     # Loss Function
#     loss_fn = [frame_loss_bce, clip_loss_bce]
#
#     # Compile the model
#     model.compile(optimizer=opt,
#                   loss=loss_fn,
#                   loss_weights=inputs.loss_weights,
#                   metrics=METRICS)
#     return model


# SED Models
def no_sed_model(inputs):
    METRICS = [LWLRAP(24),
               tfa.metrics.F1Score(num_classes=24, average='micro', name='f1'),
               ]

    # height = inputs.height
    # width = inputs.width

    # Select model
    if inputs.model == 'ResNet34':
        height, width = 224, 224
        base_model = ResNet34(input_shape=(height, width, 3),
                              include_top=False,
                              weights='imagenet')
    elif inputs.model == 'ResNet18':
        height, width = 224, 224
        base_model = ResNet18(input_shape=(height, width, 3),
                              include_top=False,
                              weights='imagenet')
    elif inputs.model == 'EfficientNetB0':
        height, width = 224, 224
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(height, width, 3),
                                                          include_top=False,
                                                          weights='imagenet')
    elif inputs.model == 'EfficientNetB3':
        height, width = 300, 300
        base_model = tf.keras.applications.EfficientNetB3(input_shape=(height, width, 3),
                                                          include_top=False,
                                                          weights='imagenet')
    elif inputs.model == 'MobileNetV2':
        height, width = 224, 224
        base_model = tf.keras.applications.MobileNetV2(input_shape=(height, width, 3),
                                                       include_top=False,
                                                       weights='imagenet')
    elif inputs.model == 'MobileNetV3Large':
        height, width = 224, 224
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=(height, width, 3),
                                                            include_top=False,
                                                            weights='imagenet')
    elif inputs.model == 'DenseNet121':
        height, width = 224, 224
        base_model = tf.keras.applications.DenseNet121(input_shape=(height, width, 3),
                                                       include_top=False,
                                                       weights='imagenet')
    elif inputs.model == 'Xception':
        height, width = 299, 299
        base_model = tf.keras.applications.Xception(input_shape=(height, width, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    elif inputs.model == 'VGG16':
        height, width = 224, 224
        base_model = tf.keras.applications.VGG16(input_shape=(height, width, 3),
                                                 include_top=False,
                                                 weights='imagenet')
    elif inputs.model == 'VGG19':
        height, width = 224, 224
        base_model = tf.keras.applications.VGG16(input_shape=(height, width, 3),
                                                 include_top=False,
                                                 weights='imagenet')

    for layer in base_model.layers:
        if inputs.base_trainable:
            # print(f'Base Layers are Trainable')
            layer.trainable = inputs.base_trainable
        else:
            # print(f'Base Layers are NOT Trainable')
            layer.trainable = False

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    if inputs.fc_layer is not None:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(inputs.fc_layer['dropout0'])(x)
        x = tf.keras.layers.Dense(inputs.fc_layer['num_hidden'], activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(inputs.fc_layer['dropout1'])(x)
    # else:
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     # x = tf.keras.layers.Dense(1280, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(0.4)(x)

    if inputs.bias_initializer is None:
        output = tf.keras.layers.Dense(24, activation='sigmoid')(x)
    else:
        output = tf.keras.layers.Dense(24, activation='sigmoid',
                                       bias_initializer=tf.keras.initializers.Constant(inputs.bias_initializer))(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimization
    opt = tf.keras.optimizers.Adam()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # loss_fn = [masked_loss_function]

    # Compile the model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=METRICS)
    return model


# Model Callbacks
def model_callbacks(x_shape, batch_size, monitor, save_h5_path, inputs):
    # Monitor
    es_monitor = monitor['early_stop']
    cp_monitor = monitor['check_point']

    if 'loss' in cp_monitor:
        cp_mode = 'min'
    if 'lwlrap' in cp_monitor:
        cp_mode = 'max'
    if 'f1' in cp_monitor:
        cp_mode = 'max'

    if 'loss' in es_monitor:
        es_mode = 'min'
    if 'lwlrap' in es_monitor:
        es_mode = 'max'
    if 'f1' in cp_monitor:
        es_mode = 'max'

    """ Select Learning Rate Scheduler """
    # steps_per_epoch = int((x_shape[0] / batch_size))
    steps_per_epoch = int(math.ceil(x_shape[0] / batch_size))
    # One Cycle Learning
    if inputs.lr_schedule['type'] == 'clr':
        clr_step_size = int(inputs.lr_schedule['clr_step'] * steps_per_epoch)
        base_lr = inputs.lr_min
        max_lr = inputs.lr_max
        clr_mode = inputs.lr_schedule['mode']
        lr_update = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=clr_mode)
        if clr_mode == 'exp_range':
            gamma = 1.0
            lr_update = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=clr_mode, gamma=gamma)

    # Cosine Annealing
    if inputs.lr_schedule['type'] == 'sgd':
        lr_update = SGDRScheduler(min_lr=inputs.lr_min,
                                  max_lr=inputs.lr_max,
                                  steps_per_epoch=steps_per_epoch,
                                  lr_decay=inputs.lr_schedule['lr_decay'],
                                  cycle_length=inputs.lr_schedule['cycle_length'],
                                  mult_factor=inputs.lr_schedule['mult_factor'])

    # Cosine Warm Up with Restarts
    if inputs.lr_schedule['type'] == 'cwu':
        warmup_epochs = int(inputs.epochs * inputs.lr_schedule['warmup_ratio'])
        lr_update = WarmUpCosineDecayScheduler(learning_rate_base=inputs.lr_schedule['learning_rate'],
                                               total_steps=int(inputs.epochs * steps_per_epoch),
                                               warmup_learning_rate=inputs.lr_schedule['warmup_learning_rate'],
                                               warmup_steps=int(warmup_epochs * steps_per_epoch),
                                               hold_base_rate_steps=0)

    class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
        """Learning rate scheduler which sets the learning rate according to schedule.

      Arguments:
          schedule: a function that takes an epoch index
              (integer, indexed from 0) and current learning rate
              as inputs and returns a new learning rate as output (float).
      """

        def __init__(self, schedule):
            super(CustomLearningRateScheduler, self).__init__()
            self.schedule = schedule

        def on_epoch_begin(self, epoch, logs=None):
            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            # Call schedule function to get the scheduled learning rate.
            scheduled_lr = self.schedule(epoch, lr)
            # Set the value back to the optimizer before this epoch starts
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

    def lr_schedule(epoch, lr):
        print(f'Learning Rate {lr}')
        return lr

    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_h5_path,
        save_weights_only=True,
        monitor=cp_monitor,
        mode=cp_mode,
        save_best_only=True,
        verbose=1,
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor=es_monitor,
        mode=es_mode,
        patience=inputs.epochs,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [
        [],
        [
            CustomLearningRateScheduler(lr_schedule),
            lr_update,
            es,
            check_point]
    ]

    return callbacks
