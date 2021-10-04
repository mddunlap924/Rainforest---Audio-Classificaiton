# Audio-Classification
This is code I developed for the [Kaggle Rainforest Connection Species Audio Detection Competition](https://www.kaggle.com/c/rfcx-species-audio-detection/overview). This is a baseline model used in the competition. With further modifications  and ensembling I was able to finish in the [top 5%](https://www.kaggle.com/dunlap0924) of this competition. This project was coded using [TensorFlow](https://www.tensorflow.org/). Furthermore, this code offers custom approaches for executing multiple models and experiments in sequential order, training for multiple folds of data, maintaining oof predictions, and custom loss functions. 

The baseline approach for this competition was to use a [Sound Event Detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection) model architecture for spectrogram images. 

The objective of this competition was to identify 24 classes of species from audio recordings. The approach was to convert audio recordings into [Mel Spectrograms](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html) and then use image classification models offered from TensorFlow for the classification task. A Mel spectrogram from an audio clip within the dataset is shown below; this spectrogram shows a few different species as determined by the various visible sound patterns.

![](https://github.com/mddunlap924/Rainforest---Audio-Classificaiton/blob/main/Images/Mel_Spectrogram.png)

In this repository an one-dimensional convolutional neural network (1D-CNN) is used to assign a probability from 0-1 for the detection of the chirp waveform. We ensemble this 1D-CNN with other two-dimensional (2D) spectrogram image classification techniques to boost our score. I provide a 2D CQT transform approach [here](https://github.com/mddunlap924/G2Net_Spectrogram-Classification) which uses image classification network architecture(s).

# Requirements

The data is ~62GB and can be found on [Kaggle](https://www.kaggle.com/c/rfcx-species-audio-detection/data). No data has been loaded to this repository.

# Sound Event Detection Model

A Sound Event Detection (SED) [[1][1], [2][2]] task predicts the class and location of the class within an audio clip. The following image illustrates a SED task and was taken from [[1][1]].

![](https://github.com/mddunlap924/Rainforest---Audio-Classificaiton/tree/main/Images/SED.png)

A SED model requires a custom architecture and loss function as provided in [model_setup_sed.py]().

```
  for layer in base_model.layers:
        if inputs.base_trainable:
            layer.trainable = inputs.base_trainable
        else:
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
```

Users are encouraged to modify the files as they see fit to best work with their applications. 

# References

[1] [https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection][1]

[2] [https://arxiv.org/abs/2107.05463][2]

[1]: https://arxiv.org/abs/2107.05463
[2]: https://arxiv.org/abs/2107.05463

