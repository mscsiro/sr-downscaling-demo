"""
Recurrent Super-Resolution Network Architecture for State Variable Downscaling
------------------------------------------------------------------------------

Date: 2023-07-31
Affiliations:
    - The University of Adelaide
    - IFP Energies nouvelles (IFPEN)

Description:
    This script defines the architecture and loss functions for a recurrent
    super-resolution convolutional neural network designed to downscale 
    coarse-scale reservoir simulation outputs to fine-scale approximations.

    The network:
        • Accepts multiple fine-scale and coarse-scale state variables 
          (e.g., pressure, saturation) and permeability fields as inputs.
        • Integrates spatial information through multiple 3D convolutional 
          and transposed convolutional layers.
        • Combines feature maps from separate branches to enhance detail 
          restoration lost during upscaling.
        • Utilises a clipped ReLU activation (max value = 1) for bounded outputs.

    Loss and Metrics:
        • Structural Similarity Index (MS-SSIM) loss to capture perceptual 
          similarity between predicted and reference fine-scale fields.
        • R² score and RMSE as additional performance metrics.

    References:
        Sayyafzadeh, M., Bouquet, S. & Gervais, V., 2024, September. Downscaling 
        State Variables of Reactive Transport Simulation Using Recurrent 
        Super-Resolution Networks. In ECMOR 2024 (Vol. 2024, No. 1, pp. 1-18). 
        European Association of Geoscientists & Engineers.

Usage:
    The `model_create_two_state` function can be imported and called within 
    the main training script. It returns a compiled Keras model ready for 
    training with the specified input dimensions and convolution kernel sizes.
"""
# M Sayyaf 2023/07/31
import numpy as np
import tensorflow as tf

from tensorflow.keras.activations import relu


clipped_relu = lambda x: relu(x, max_value=1)


def r_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


def ssim_loss(y_true, y_pred):
    l = np.zeros((np.shape(y_true)[3],), dtype=object)
    for k in range(np.shape(y_true)[3]):
        y_true_l = y_true[:, :, :, k, :]
        y_pred_l = y_pred[:, :, :, k, :]
        l[k] = tf.reduce_mean(tf.image.ssim_multiscale(y_true_l, y_pred_l, max_val=1.0, filter_size=5))
    return 1 - np.sum(l)/np.shape(y_true)[3]



def model_create_two_state(x_fine, y_fine, z_fine, x_coarse, y_coarse, z_coarse, ks_x, ks_y, ks_z):
    ks = (ks_x, ks_y, ks_z)
    x_shape = np.ones((1, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
    input_s1 = tf.keras.Input(shape=(np.shape(x_shape)[1:]))
    input_s2 = tf.keras.Input(shape=(np.shape(x_shape)[1:]))
    input_s1_p = tf.keras.Input(shape=(np.shape(x_shape)[1:]))
    # input_s1_p_noise = tf.keras.layers.GaussianNoise(0.0001)(input_s1_p)
    input_s2_p = tf.keras.Input(shape=(np.shape(x_shape)[1:]))
    # input_s2_p_noise = tf.keras.layers.GaussianNoise(0.0001)(input_s2_p)
    input_perm = tf.keras.Input(shape=(np.shape(x_shape)[1:]))
    input_ds1 = tf.keras.Input(shape=(np.shape(x_shape)[1:]))

    input_all = tf.keras.layers.Concatenate()([input_perm, input_s1, input_s2, input_s1_p, input_s2_p, input_ds1])

    x_1 = tf.keras.layers.Conv3D(50, ks, strides=(1, 1, 1),  use_bias=True, padding='same',
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1))(input_s1)
    x_2 = tf.keras.layers.Conv3D(100, ks,  strides=(1, 1, 1),  use_bias=True,padding='same',
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1))(input_all)
    x_3 = tf.keras.layers.Conv3D(100, ks, strides=(1, 1, 1),  use_bias=True, padding='same',
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x_2)
    x_d = tf.keras.layers.Conv3D(1, ks,  strides=(1, 1, 1),  use_bias=True,padding='same',
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x_3)

    x_d = tf.keras.layers.Flatten()(x_d)
    x_d = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=True)(x_d)
    x_d = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=True)(x_d)
    x_d = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=True)(x_d)
    x_d = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=True)(x_d)
    x_d = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=True)(x_d)
    x_d = tf.keras.layers.Dense(x_coarse*y_coarse*z_coarse, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                use_bias=True)(x_d)
    x_d = tf.keras.layers.Reshape((x_coarse, y_coarse, z_coarse, 1))(x_d)
    x_d = tf.keras.layers.Dropout(0.01)(x_d)
    x_d = tf.keras.layers.Conv3DTranspose(100, (int(x_fine/x_coarse), int(y_fine/y_coarse), int(z_fine/z_coarse)),
                                          strides=(int(x_fine/x_coarse), int(y_fine/y_coarse), int(z_fine/z_coarse)),
                                          padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x_d)
    x_m = tf.keras.layers.Concatenate()([x_1, x_3, x_d])
    y_2 = tf.keras.layers.Conv3D(100, ks,   strides=(ks_x, ks_y, 1), padding='same',
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x_m)
    y_2 = tf.keras.layers.Conv3DTranspose(100, ks,   strides=(ks_x, ks_y, 1),
                                          padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(y_2)
    y = tf.keras.layers.Concatenate()([x_2, y_2])
    y = tf.keras.layers.Conv3D(50, ks,   strides=(1, 1, 1), padding='same', activation='relu')(y)
    y = tf.keras.layers.Conv3D(50, ks,   strides=(1, 1, 1), padding='same', activation='relu')(y)
    y = tf.keras.layers.Conv3D(50, ks,   strides=(1, 1, 1), padding='same', activation='relu')(y)
    y = tf.keras.layers.Conv3D(1, (1, 1, 1),   strides=(1, 1, 1), padding='same', activation=clipped_relu)(y)
    # y = tf.keras.layers.Conv3D(1, (1, 1, 1),   strides=(1, 1, 1), padding='same', activation='relu')(y)

    model = tf.keras.Model([input_perm, input_s1, input_s2,  input_s1_p, input_s2_p, input_ds1], outputs=y)

    model.compile(loss=ssim_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1),
                  metrics=['acc', tf.keras.metrics.RootMeanSquaredError(), r_squared])
    return model


