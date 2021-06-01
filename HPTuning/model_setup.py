import tensorflow as tf
import tensorflow_addons as tfa
from lwlrap import LWLRAP
from clr_callback import CyclicLR
from sgd_callback import SGDRScheduler
from classification_models.tfkeras import Classifiers
import numpy as np

ResNet34, _ = Classifiers.get('resnet34')
ResNet18, _ = Classifiers.get('resnet18')

# METRICS = [[tf.keras.metrics.BinaryCrossentropy(name='seg_bc_loss'),
#            tfa.metrics.F1Score(num_classes=24, average='micro', name='seg_f1'),
#            LWLRAP(24)],
# [tf.keras.metrics.BinaryCrossentropy(name='clip_bc_loss'),
#            tfa.metrics.F1Score(num_classes=24, average='micro', name='clip_f1'),
#            LWLRAP(24)]]

METRICS = [tf.keras.metrics.BinaryCrossentropy(name='bc_loss'),
           tfa.metrics.F1Score(num_classes=24, average='micro', name='f1'),
           LWLRAP(24)]


# ResNet Models
def resnet_model(inputs):
    height = inputs.height
    width = inputs.width

    if isinstance(inputs.output_bias, float):
        output_bias = tf.keras.initializers.Constant(inputs.output_bias)

    # Select model
    if inputs.model == 'ResNet34':
        base_model = ResNet34(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'ResNet18':
        base_model = ResNet18(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB3':
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet')

    if inputs.model == 'MobileNetV2':
        input_shape = (inputs.height, inputs.width, 3)
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # for layer in base_model.layers[:0]:
    #     layer.trainable = False

    for layer in base_model.layers:
        layer.trainable = True

    # Global Pooling
    x0 = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x1 = tf.keras.layers.GlobalMaxPooling2D()(base_model.output)

    # Select Pooling Method
    if inputs.layer_combination == 'Avg':
        x = x0
    elif inputs.layer_combination == 'Max':
        x = x1
    elif inputs.layer_combination == 'AvgMax':
        x = (x0 + x1) / 2

    # Form remainder of model
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(inputs.dropout0)(x)
    x = tf.keras.layers.Dense(inputs.num_hidden, activation='relu',
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              kernel_regularizer=tf.keras.regularizers.l2(inputs.regularization))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(inputs.dropout1)(x)
    if inputs.output_bias is None:
        output = tf.keras.layers.Dense(24, activation="sigmoid", name="output")(x)
    else:
        output = tf.keras.layers.Dense(24, activation="sigmoid", bias_initializer=output_bias, name="output")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimization
    opt = tf.keras.optimizers.Adam()

    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=METRICS)
    return model


# SED Models
def sed_model(inputs):
    height = inputs.height
    width = inputs.width
    data_format = 'channels_last'
    if isinstance(inputs.output_bias, float):
        output_bias = tf.keras.initializers.Constant(inputs.output_bias)

    # Select model
    if inputs.model == 'ResNet34':
        base_model = ResNet34(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'ResNet18':
        base_model = ResNet18(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB3':
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet')

    if inputs.model == 'MobileNetV2':
        input_shape = (inputs.height, inputs.width, 3)
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # for layer in base_model.layers[:0]:
    #     layer.trainable = False

    for layer in base_model.layers:
        layer.trainable = True

    # Aggregate over Frequency Axis
    x = tf.reduce_mean(base_model.output, axis=1)
    # (Batch Size, Time, Filters)

    # Max and Avg. Over the Filter Axis
    x1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x2 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = x1 + x2
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    # (Batch Size, Time, Filters)

    # Classifier / Tagging / Segmentwise
    # segmentwise_output = tf.keras.layers.Dense(24, activation="sigmoid", name="segmentwise_output")(x)
    segmentwise_output = tf.keras.layers.Conv1D(filters=24, kernel_size=1, padding='same', activation="sigmoid",
                                                name="segmentwise_output", data_format=data_format)(x)

    # Attention Module / Class Probability with Softmax
    #  norm_att = tf.keras.layers.Dense(24, name='start_norm_att')(x)
    norm_att = tf.keras.layers.Conv1D(24, kernel_size=1, padding='same', name='start_norm_att',
                                      data_format=data_format)(x)
    #  norm_att = tf.keras.layers.BatchNormalization()(norm_att)
    norm_att = tf.keras.activations.tanh(norm_att / 10) * 10
    norm_att = tf.keras.activations.softmax(norm_att, axis=-2, name='end_norm_att')
    # axis=-2 sum each class to 1 - i.e. each 24 bird class sums to 1
    # axis=-1 sum each time window to 1 - i.e. 16 values where each has a sum of 1
    # axis=0 sum of entire matrix is 1 - i.e. sum of all values in 2D matrix is 1

    # Clipwise output
    clipwise_output = tf.math.reduce_sum(norm_att * segmentwise_output, axis=1)
    # clipwise_output = tf.keras.activations.sigmoid(clipwise_output, name='clipwise_activation')
    clipwise_output = tf.keras.layers.Lambda(lambda x: x, name="clipwise_output")(clipwise_output)
    output = clipwise_output

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimization
    opt = tf.keras.optimizers.Adam()

    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=METRICS)
    return model


# SED Models
def sed_model2(inputs):
    METRICS = [[tf.keras.metrics.BinaryCrossentropy(name='seg_bc_loss'),
                tfa.metrics.F1Score(num_classes=24, average='macro', name='seg_f1'),
                LWLRAP(24)],
               [tf.keras.metrics.BinaryCrossentropy(name='bc_loss'),
                tfa.metrics.F1Score(num_classes=24, average='macro', name='f1'),
                LWLRAP(24)]]

    height = inputs.height
    width = inputs.width
    data_format = 'channels_last'
    if isinstance(inputs.output_bias, float):
        output_bias = tf.keras.initializers.Constant(inputs.output_bias)

    # Select model
    if inputs.model == 'ResNet34':
        base_model = ResNet34(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'ResNet18':
        base_model = ResNet18(input_shape=(height, width, 3), include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')

    if inputs.model == 'EfficientNetB3':
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet')

    if inputs.model == 'MobileNetV2':
        input_shape = (inputs.height, inputs.width, 3)
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # for layer in base_model.layers[:0]:
    #     layer.trainable = False

    for layer in base_model.layers:
        layer.trainable = True

    # Aggregate over Frequency Axis
    x = tf.reduce_mean(base_model.output, axis=1)
    # (Batch Size, Time, Filters)

    # Max and Avg. Over the Filter Axis
    x1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x2 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = x1 + x2
    x = x2
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # (Batch Size, Time, Filters)

    # Classifier / Tagging / Segmentwise
    # segmentwise_output = tf.keras.layers.Dense(24, activation="sigmoid", name="segmentwise_output")(x)
    segmentwise_output = tf.keras.layers.Conv1D(filters=24, kernel_size=1, padding='same', activation="sigmoid",
                                                name="segmentwise_output", data_format=data_format)(x)

    # Attention Module / Class Probability with Softmax
    # norm_att = tf.keras.layers.Dense(24, name='start_norm_att')(x)
    norm_att = tf.keras.layers.Conv1D(24, kernel_size=1, padding='same', name='start_norm_att',
                                       data_format=data_format)(x)
    norm_att = tf.keras.layers.BatchNormalization()(norm_att)
    norm_att = tf.keras.activations.tanh(norm_att / 10) * 10
    norm_att = tf.keras.activations.softmax(norm_att, axis=-2, name='end_norm_att')
    # axis=-2 sum each class to 1 - i.e. each 24 bird class sums to 1
    # axis=-1 sum each time window to 1 - i.e. 16 values where each has a sum of 1
    # axis=0 sum of entire matrix is 1 - i.e. sum of all values in 2D matrix is 1

    # Clipwise output
    clipwise_output = tf.math.reduce_sum(norm_att * segmentwise_output, axis=1)
    # clipwise_output = tf.keras.activations.sigmoid(clipwise_output, name='clipwise_activation')
    clipwise_output = tf.keras.layers.Lambda(lambda x: x, name="clipwise_output")(clipwise_output)
    output = [tf.math.reduce_max(segmentwise_output, axis=1), clipwise_output]

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Optimization
    opt = tf.keras.optimizers.Adam()

    # Loss Function
    loss_fn = [tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.BinaryCrossentropy()]

    # Compile the model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  loss_weights=[0.5, 1.0],
                  metrics=METRICS)
    return model


# Model Callbacks
def model_callbacks(x_shape, batch_size, monitor, save_h5_path, inputs):
    # Monitor
    es_monitor = monitor['early_stop']
    cp_monitor = monitor['check_point']

    if cp_monitor == "val_bc_loss":
        cp_mode = 'min'
    if cp_monitor == "val_loss":
        cp_mode = 'min'
    if cp_monitor == "val_lwlrap":
        cp_mode = 'max'
    if cp_monitor == "val_f1":
        cp_mode = 'max'

    if es_monitor == "val_bc_loss":
        es_mode = 'min'
    if es_monitor == "val_loss":
        es_mode = 'min'
    if es_monitor == "val_lwlrap":
        es_mode = 'max'
    if es_monitor == "val_f1":
        es_mode = 'max'

    # One Cycle Learning
    clr_step_size = int(inputs.clr_step * (x_shape[0] / batch_size))
    base_lr = inputs.lr_min
    max_lr = inputs.lr_max
    clr_mode = inputs.clr_mode
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=clr_mode)

    sgd = SGDRScheduler(min_lr=5e-6,
                        max_lr=5e-3,
                        steps_per_epoch=int(x_shape[0] / batch_size),
                        lr_decay=0.8,
                        cycle_length=5,
                        mult_factor=1.5)

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
        patience=inputs.patience,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [CustomLearningRateScheduler(lr_schedule),
                 # clr,
                 sgd,
                 es, check_point]

    callbacks = [[],
                 [CustomLearningRateScheduler(lr_schedule),
                 # clr,
                 sgd,
                 es, check_point]]

    return callbacks
