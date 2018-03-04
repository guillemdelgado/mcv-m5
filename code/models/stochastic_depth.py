import tensorflow as tf
from keras.layers import Input, Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, BatchNormalization, merge
from keras.models import Model
import keras.backend as K


def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise


def zero_pad_channels(x, pad=0):
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def stochastic_survival(y, p_survival=1.0):
    survival = K.random_binomial((1,), p=p_survival)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y,
                           survival * y)


def stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1):
    """
    Residual block consisting of:
    - Conv - BN - ReLU - Conv - BN
    - identity shortcut connection
    - merge Conv path with shortcut path
    """

    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
        if nb_filters > prev_nb_channels:
            shortcut = Lambda(zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
    else:
        subsample = (1, 1)
        shortcut = x

    y = Conv2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)

    p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(y)

    out = merge([y, shortcut], mode='sum')
    return out

def build_stochastic_depth_nn(img_shape=(224, 224, 3), n_classes=1000, blocks_per_group=33):
    inputs = Input(shape=img_shape)

    x = Conv2D(16, 3, 3,
                      init='he_normal', border_mode='same', dim_ordering='tf')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(0, blocks_per_group):
        nb_filters = 16
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=i, nb_total_blocks=3 * blocks_per_group,
                                            subsample_factor=1)

    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=blocks_per_group + i, nb_total_blocks=3 * blocks_per_group,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=3 * blocks_per_group,
                                            subsample_factor=subsample_factor)

    x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
    x = Flatten()(x)

    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    return model

