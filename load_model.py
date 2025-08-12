# A general network for downscaling state variables M Sayyaf 2023 Jul
"""
Recurrent Super-Resolution Downscaling (Demo)
----------------------------------------------
This script is part of the demonstration workflow for coarse-to-fine
downscaling of subsurface flow simulations using a recurrent
super-resolution convolutional neural network.

Affiliations:
    - The University of Adelaide
    - IFP Energies Nouvelles (IFPEN)

Data provenance:
    - Multiple-Point Geostatistics training image (Strebelle, 2002)
    - MATLAB Reservoir Simulation Toolbox (MRST)
    - OPM Flow simulator

License:
    MIT License (see LICENSE file in the repository root)

Citation:
If you use this repository, please cite the associated publication:

Sayyafzadeh, M., Bouquet, S., & Gervais, V. (2024, September).  
**Downscaling State Variables of Reactive Transport Simulation Using Recurrent Super-Resolution Networks.**  
In *ECMOR 2024* (Vol. 2024, No. 1, pp. 1–18).  
European Association of Geoscientists & Engineers.  
https://doi.org/10.3997/2214-4609.202452072


File purpose:
    This script loads fine- and coarse-scale simulation data, prepares
    inputs/targets for network testing and plot the results (for saturation and pressure).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
from matplotlib import colors
import matplotlib
import random
from base_model import r_squared, ssim_loss

seed_no = 3
tf.keras.utils.set_random_seed(seed_no)  # sets seeds for base-python, numpy and tf
tf.config.experimental.enable_op_determinism()


def R_squared_map(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)), 0)
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y, 0))), 0)
    r2_map = tf.subtract(1.0, tf.divide(residual, total))
    return r2_map


def mean_sq_map(y, y_pred):
    residual = tf.reduce_mean(tf.square(tf.subtract(y, y_pred)), 0)
    return residual

Train_p = True
n_test_calc = 20

loc_x = [158, 80]
loc_y = [158, 80]
z_layer = [1, 1]
time_steps = np.arange(24, 121, 12, dtype=int)

def ensure_5d_xyztn(a, name):
    a = np.asarray(a)
    if a.ndim == 5:
        return a
    if a.ndim == 4:
        # MATLAB coarse case often drops z=1; insert at axis=2
        a = np.expand_dims(a, axis=2)
        return a
    raise ValueError(f"{name}: unexpected shape {a.shape}; expected 4D or 5D")


path_dir = "samples/"
model_name = "strabelle"

# Input MAT file names (without extension)
file_name_sw_fine = model_name + "_fine_all_sw"      # Fine-scale saturation
file_name_p_fine = model_name + "_fine_all_p"        # Fine-scale pressure
file_name_sw_coarse = model_name + "_coarse_all_sw"  # Coarse-scale saturation
file_name_p_coarse = model_name + "_coarse_all_p"    # Coarse-scale pressure
file_name_perm_fine = model_name + "_fine_all_perm"  # Fine-scale permeability


# Fine and coarse grid sizes
x_fine, y_fine, z_fine = 159, 159, 2
x_coarse, y_coarse, z_coarse = 53, 53, 1

# Training configuration
n_train = 80          # Number of training realisations
n_tsteps = 9          # Number of time steps per realisation (input-output pairs)
n_epochs = 200
batch_size = 5

# Normalisation constants
max_p, min_p = 1000, 100
max_perm = 1000
max_sw, min_sw = 1, 0

# Kernel sizes for network architecture
ks_x = np.min([10, int(x_fine / x_coarse)])
ks_y = np.min([10, int(y_fine / y_coarse)])
ks_z = np.max([2, int(z_fine / z_coarse)])

mat_con = mat73.loadmat(path_dir + file_name_sw_fine + '.mat')
Y_sW = mat_con["Y_sW"]
Y_sW = ensure_5d_xyztn(Y_sW, "Y_sW")   # expect (159,159,2,10,100)

Y_sW_n = np.zeros((np.shape(Y_sW)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
for n in range(np.shape(Y_sW)[-1]):
    for t in range(n_tsteps):
        for k in range(z_fine):
            Y_sW_n[n*n_tsteps+t, :, :, k, 0] = Y_sW[:, :, k, t+1, n]/max_sw

X_sW_n_p = np.zeros((np.shape(Y_sW)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
for n in range(np.shape(Y_sW)[-1]):
    for t in range(n_tsteps):
        for k in range(z_fine):
            X_sW_n_p[n * n_tsteps + t, :, :, k, 0] = Y_sW[:, :, k, t, n]/max_sw


# -----------------------------------------------------------------------------
# 4. Load and normalise fine-scale pressure data
# -----------------------------------------------------------------------------
mat_con = mat73.loadmat(path_dir + file_name_p_fine + '.mat')
Y_p = mat_con["Y_p"]
Y_p  = ensure_5d_xyztn(Y_p,  "Y_p")    # expect (159,159,2,10,100)

Y_p_n = np.zeros((np.shape(Y_p)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
for n in range(np.shape(Y_p)[-1]):
    for t in range(n_tsteps):
        for k in range(z_fine):
            Y_p_n[n*n_tsteps+t, :, :, k, 0] = (Y_p[:, :, k, t+1, n]-min_p)/(max_p-min_p)

X_p_n_p = np.ones((np.shape(Y_p)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
for n in range(np.shape(Y_p)[-1]):
    for t in range(n_tsteps):
        for k in range(z_fine):
            X_p_n_p[n * n_tsteps + t, :, :, k, 0] = (Y_p[:, :, k, t, n]-min_p)/(max_p-min_p)


# -----------------------------------------------------------------------------
# 5. Load coarse-scale saturation and compute deltas
# -----------------------------------------------------------------------------

mat_con = mat73.loadmat(path_dir + file_name_sw_coarse + '.mat')
X_sW = mat_con["X_sW"]
X_sW = ensure_5d_xyztn(X_sW, "X_sW")   # expect (53,53,1,10,100)
X_sW_n = np.zeros((np.shape(X_sW)[-1]*n_tsteps, x_coarse, y_coarse, z_coarse, 1))
X_dsW_n = np.zeros((np.shape(X_sW)[-1]*n_tsteps, x_coarse, y_coarse, z_coarse, 1))
for n in range(np.shape(X_sW)[-1]):
    for t in range(n_tsteps):
        for k in range(z_coarse):
            X_sW_n[n*n_tsteps+t, :, :, k, 0] = X_sW[:, :, k, t+1, n]/max_sw
            X_dsW_n[n*n_tsteps+t, :, :, k, 0] = X_sW[:, :, k, t+1, n]/max_sw - X_sW[:, :, k, t, n]/max_sw

# Upsample coarse fields to fine resolution for input to network
X_sW_n_re = np.zeros((np.shape(X_sW)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
X_dsW_n_re = np.zeros((np.shape(X_sW)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
for n in range(np.shape(X_sW_n)[0]):
    for k in range(z_fine):
        X_sW_n_re[n, :, :, k, :] = tf.image.resize(X_sW_n[n, :, :, 0, :], (x_fine, y_fine), method='bilinear')
        X_dsW_n_re[n, :, :, k, :] = tf.image.resize(X_dsW_n[n, :, :, 0, :], (x_fine, y_fine), method='bilinear')

mat_con = mat73.loadmat(path_dir + file_name_p_coarse + '.mat')
X_p = mat_con["X_p"]
X_p  = ensure_5d_xyztn(X_p,  "X_p")    # expect (53,53,1,10,100)
X_p_n = np.zeros((np.shape(X_p)[-1]*n_tsteps, x_coarse, y_coarse, z_coarse, 1))
X_dp_n = np.zeros((np.shape(X_p)[-1]*n_tsteps, x_coarse, y_coarse, z_coarse, 1))
for n in range(np.shape(X_p)[-1]):
    for t in range(n_tsteps):
        for k in range(z_coarse):
            X_p_n[n*n_tsteps+t, :, :, k, 0] = (X_p[:, :, k, t+1, n]-min_p)/(max_p-min_p)
            X_dp_n[n*n_tsteps+t, :, :, k, 0] = (X_p[:, :, k, t+1, n]-min_p)/(max_p-min_p) - (X_p[:, :, k, t, n]-min_p)/(max_p-min_p)


# -----------------------------------------------------------------------------
# 6. Load coarse-scale pressure and compute deltas
# -----------------------------------------------------------------------------
X_p_n_re = np.zeros((np.shape(X_p)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))
X_dp_n_re = np.zeros((np.shape(X_p)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1))

for n in range(np.shape(X_sW_n)[0]):
    for k in range(z_fine):
        X_p_n_re[n, :, :, k, :] = tf.image.resize(X_p_n[n, :, :, 0, :], (x_fine, y_fine), method='bilinear')
        X_dp_n_re[n, :, :, k, :] = tf.image.resize(X_dp_n[n, :, :, 0, :], (x_fine, y_fine), method='bilinear')


# -----------------------------------------------------------------------------
# 7. Load fine-scale permeability
# -----------------------------------------------------------------------------
file_name_perm_fine = model_name + "_fine_all_perm"
mat_con = mat73.loadmat(path_dir + file_name_perm_fine + '.mat')
X_perm = mat_con["X_PERM"]
X_perm_t = np.ones((np.shape(X_perm)[-1]*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
for n in range(np.shape(X_perm)[-1]):
    for t in range(n_tsteps):
        for k in range(z_fine):
            X_perm_t[n*n_tsteps+t, :, :, k, 0] = X_perm[:, :, k,  n]/max_perm


x_test_p_a = X_p_n_re[n_train*n_tsteps:,  :, :, :,: ]
x_test_co2_a = X_sW_n_re[n_train*n_tsteps:, :,  :, :,:]
x_test_perm = X_perm_t[n_train*n_tsteps:,  :, :, :,:]
x_test_co2_p = X_sW_n_p[n_train*n_tsteps:, :,  :, :,:]
x_test_p_p = X_p_n_p[n_train*n_tsteps:, :,  :, :,:]
y_test_p = Y_p_n[n_train*n_tsteps:, :, :, :, :]
y_test_co2 = Y_sW_n[n_train*n_tsteps:, :, :, :, :]
x_test_dco2 = X_dsW_n_re[n_train*n_tsteps:, :, :, :,:]
x_test_dp = X_dp_n_re[n_train*n_tsteps:, :, :, :,:]


model_sw = tf.keras.models.load_model('model_sw', custom_objects={'r_squared': r_squared, 'ssim_loss': ssim_loss})


model_p = tf.keras.models.load_model('model_p', custom_objects={'r_squared': r_squared, 'ssim_loss': ssim_loss})

SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

samples_c = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_f = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_r = np.zeros((n_test_calc, len(loc_x), n_tsteps))

samples_c_p = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_f_p = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_r_p = np.zeros((n_test_calc, len(loc_x), n_tsteps))

samples_c_ph = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_f_ph = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_r_ph = np.zeros((n_test_calc, len(loc_x), n_tsteps))

samples_c_deltaphi = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_f_deltaphi = np.zeros((n_test_calc, len(loc_x), n_tsteps))
samples_r_deltaphi = np.zeros((n_test_calc, len(loc_x), n_tsteps))

r2_p = np.zeros((n_test_calc,n_tsteps))
r2_co2 = np.zeros((n_test_calc,n_tsteps))
r2_ph = np.zeros((n_test_calc,n_tsteps))
r2_deltaphi = np.zeros((n_test_calc,n_tsteps))

e2_p = np.zeros((n_test_calc,n_tsteps))
e2_co2 = np.zeros((n_test_calc,n_tsteps))
e2_ph = np.zeros((n_test_calc,n_tsteps))
e2_deltaphi = np.zeros((n_test_calc,n_tsteps))

y_test_p_ap_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_co2_ap_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_ph_ap_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_deltaphi_ap_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))

y_test_p_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_co2_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_ph_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))
y_test_deltaphi_all_lasttimestep = np.zeros((n_test_calc, x_fine, y_fine, z_fine, 1))


for i in range(n_test_calc):
    s_c = np.zeros((len(loc_x), n_tsteps))
    s_f = np.zeros((len(loc_x), n_tsteps))
    s_r = np.zeros((len(loc_x), n_tsteps))
    p_c = np.zeros((len(loc_x), n_tsteps))
    p_f = np.zeros((len(loc_x), n_tsteps))
    p_r = np.zeros((len(loc_x), n_tsteps))

    y_test_p_ap = model_p.predict(
        [x_test_perm[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
            x_test_p_a[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
            x_test_co2_a[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
            x_test_p_p[(n_tsteps * i):(i * n_tsteps + 1), :, :, :],
            x_test_co2_p[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
            x_test_dp[(n_tsteps*i):(i*n_tsteps+1), : ,: , :]])

    # y_test_co2_ap = y_test_co2[(n_tsteps * i):(i * n_tsteps + 1), :, :, :]
    y_test_co2_ap = model_sw.predict(
        [x_test_perm[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
         x_test_co2_a[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
         x_test_p_a[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
         x_test_co2_p[(n_tsteps*i):(i*n_tsteps+1), : ,:, :],
         x_test_p_p[(n_tsteps*i):(i*n_tsteps+1), : ,: , :],
         x_test_dco2[(n_tsteps*i):(i*n_tsteps+1), : ,: , :]])


    r2_p[i, 0] = r_squared(y_test_p[(n_tsteps*i):(i*n_tsteps+1), :, :, :], y_test_p_ap)
    r2_co2[i, 0] = r_squared(y_test_co2[(n_tsteps*i):(i*n_tsteps+1), :, :, :], y_test_co2_ap)

    e2_p[i, 0] = np.mean(tf.keras.metrics.mean_squared_error(y_test_p[(n_tsteps*i):(i*n_tsteps+1), :, :, :], y_test_p_ap).numpy())
    e2_co2[i, 0] = np.mean(tf.keras.metrics.mean_squared_error(y_test_co2[(n_tsteps*i):(i*n_tsteps+1), :, :, :], y_test_co2_ap).numpy())

    for nw in range(len(loc_x)):
        s_f[nw, 0] = y_test_co2[i * n_tsteps, loc_x[nw], loc_y[nw], z_layer[nw], 0]
        s_c[nw, 0] = x_test_co2_a[i * n_tsteps, loc_x[nw], loc_y[nw], z_layer[nw], 0]
        s_r[nw, 0] = y_test_co2_ap[0, loc_x[nw], loc_y[nw], z_layer[nw], 0]

        p_f[nw, 0] = y_test_p[i * n_tsteps, loc_x[nw], loc_y[nw], z_layer[nw], 0]
        p_c[nw, 0] = x_test_p_a[i * n_tsteps, loc_x[nw], loc_y[nw], z_layer[nw], 0]
        p_r[nw, 0] = y_test_p_ap[0, loc_x[nw], loc_y[nw], z_layer[nw], 0]


    for t in range(1, n_tsteps):
        y_test_p_ap_temp = model_p.predict(
            [x_test_perm[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :],
                x_test_p_a[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :],
                x_test_co2_a[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :],
                y_test_p_ap,
                y_test_co2_ap,
                x_test_dp[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:]])

        # y_test_co2_ap_temp = y_test_co2[(n_tsteps * i + t):(i * n_tsteps + t + 1), :, :, :, :]
        y_test_co2_ap_temp = model_sw.predict(
            [x_test_perm[(n_tsteps * i + t):(i * n_tsteps + t + 1), :, :, :],
             x_test_co2_a[(n_tsteps * i + t):(i * n_tsteps + t + 1), :, :, :],
             x_test_p_a[(n_tsteps * i + t):(i * n_tsteps + t + 1), :, :, :],
             y_test_co2_ap,
             y_test_p_ap,
             x_test_dco2[(n_tsteps * i + t):(i * n_tsteps + t + 1), :, :, :]])


        y_test_p_ap = y_test_p_ap_temp
        y_test_co2_ap = y_test_co2_ap_temp
        
        r2_p[i, t] = r_squared(y_test_p[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:], y_test_p_ap)
        r2_co2[i, t] = r_squared(y_test_co2[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:], y_test_co2_ap)

        e2_p[i, t] = np.mean(tf.keras.metrics.mean_squared_error(y_test_p[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:], y_test_p_ap).numpy())
        e2_co2[i, t] = np.mean(tf.keras.metrics.mean_squared_error(y_test_co2[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:], y_test_co2_ap).numpy())

        for nw in range(len(loc_x)):
            s_f[nw, t] = y_test_co2[i * n_tsteps+t, loc_x[nw], loc_y[nw], z_layer[nw], 0]
            s_c[nw, t] = x_test_co2_a[i * n_tsteps+t, loc_x[nw], loc_y[nw], z_layer[nw], 0]
            s_r[nw, t] = y_test_co2_ap[0, loc_x[nw], loc_y[nw], z_layer[nw], 0]
            p_f[nw, t] = y_test_p[i * n_tsteps + t, loc_x[nw], loc_y[nw], z_layer[nw], 0]
            p_c[nw, t] = x_test_p_a[i * n_tsteps + t, loc_x[nw], loc_y[nw], z_layer[nw], 0]
            p_r[nw, t] = y_test_p_ap[0, loc_x[nw], loc_y[nw], z_layer[nw], 0]

    y_test_p_ap_all_lasttimestep[i] = y_test_p_ap
    y_test_p_all_lasttimestep[i] = y_test_p[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:]
    y_test_co2_ap_all_lasttimestep[i] = y_test_co2_ap
    y_test_co2_all_lasttimestep[i] = y_test_co2[(n_tsteps*i+t):(i*n_tsteps+t+1), : ,:, :,:]

    samples_r[i, :, :] = s_r
    samples_f[i, :, :] = s_f
    samples_c[i, :, :] = s_c
    samples_r_p[i, :, :] = p_r
    samples_f_p[i, :, :] = p_f
    samples_c_p[i, :, :] = p_c

    plt.tight_layout()
    fig, axes = plt.subplots(nrows=z_fine, ncols=3)
    for k in range(z_fine):
        ax = plt.subplot(z_fine, 3, k*3 + 3)
        im = plt.imshow(y_test_co2_ap[0, :, :, k, 0].transpose(), vmin=np.min(y_test_co2[i*n_tsteps+t]), vmax=np.max(y_test_co2[i*n_tsteps+t]), origin='lower')
        plt.title("Network - layer#" + str(k+1))
        ax = plt.subplot(z_fine, 3, k*3 + 2)
        plt.title("FS simulation - layer#" + str(k+1))
        im = plt.imshow(y_test_co2[i*n_tsteps+t, :, :, k, 0].transpose(), vmin=np.min(y_test_co2[i*n_tsteps+t]), vmax=np.max(y_test_co2[i*n_tsteps+t]), origin='lower')
        ax = plt.subplot(z_fine, 3, k*3 + 1)
        im = plt.imshow(x_test_co2_a[i*n_tsteps+t, :, :, k, 0].transpose(), vmin=np.min(y_test_co2[i*n_tsteps+t]), vmax=np.max(y_test_co2[i*n_tsteps+t]), origin='lower')
        plt.title("CS simulation - layer#" + str(k+1))
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
    fig.colorbar(im, cax=cbar_ax, shrink=0.50)
    plt.savefig('results_test_co2_'+ str(i) + '.png')
    plt.close('all')

    plt.tight_layout()
    fig, axes = plt.subplots(nrows=z_fine, ncols=3)
    for k in range(z_fine):
        ax = plt.subplot(z_fine, 3, k*3 + 3)
        im = plt.imshow(y_test_p_ap[0, :, :, k, 0].transpose(), vmin=np.min(y_test_p[i*n_tsteps+t]), vmax=np.max(y_test_p[i*n_tsteps+t]), origin='lower')
        plt.title("Network - layer#" + str(k+1))
        ax = plt.subplot(z_fine, 3, k*3 + 2)
        im = plt.imshow(y_test_p[i*n_tsteps+t, :, :, k, 0].transpose(), vmin=np.min(y_test_p[i*n_tsteps+t]), vmax=np.max(y_test_p[i*n_tsteps+t]), origin='lower')
        plt.title("FS simulation - layer#" + str(k+1))
        ax = plt.subplot(z_fine, 3, k*3 + 1)
        im = plt.imshow(x_test_p_a[i*n_tsteps+t, :, :, k, 0].transpose(), vmin=np.min(y_test_p[i*n_tsteps+t]), vmax=np.max(y_test_p[i*n_tsteps+t]), origin='lower')
        plt.title("CS simulation - layer#" + str(k+1))
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
    fig.colorbar(im, cax=cbar_ax, shrink=0.50)
    plt.savefig('results_test_p_' + str(i) + '.png')
    plt.close('all')


mean_e_p_map = mean_sq_map(y_test_p_all_lasttimestep, y_test_p_ap_all_lasttimestep)
mean_e_co2_map = mean_sq_map(y_test_co2_all_lasttimestep, y_test_co2_ap_all_lasttimestep)

plt.tight_layout()
fig, axes = plt.subplots(nrows=z_fine, ncols=3)
for k in range(z_fine):
    ax = plt.subplot(1, z_fine, k + 1)
    im = plt.imshow(np.log(mean_e_p_map[:, :, k, 0]), origin='lower')
    plt.title("Layer" + str(k+1))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
fig.colorbar(im, cax=cbar_ax, shrink=0.50)
plt.savefig('er_map_p.png')
plt.close('all')

plt.tight_layout()
fig, axes = plt.subplots(nrows=z_fine, ncols=3)
for k in range(z_fine):
    ax = plt.subplot(1, z_fine, k + 1)
    im = plt.imshow(np.log(mean_e_co2_map[:, :, k, 0]), origin='lower')
    plt.title("Layer" + str(k+1))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
fig.colorbar(im, cax=cbar_ax, shrink=0.50)
plt.savefig('er_map_co2.png')
plt.close('all')



r2_p_map = tf.clip_by_value(R_squared_map(y_test_p_all_lasttimestep, y_test_p_ap_all_lasttimestep), -1, 1)
r2_co2_map = tf.clip_by_value(R_squared_map(y_test_co2_all_lasttimestep, y_test_co2_ap_all_lasttimestep), -1, 1)

plt.tight_layout()
fig, axes = plt.subplots(nrows=z_fine, ncols=3)
for k in range(z_fine):
    ax = plt.subplot(1, z_fine, k + 1)
    im = plt.imshow(r2_p_map[:, :, k, 0], origin='lower')
    plt.title("Layer" + str(k+1))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
fig.colorbar(im, cax=cbar_ax, shrink=0.50)
plt.savefig('r2_map_p.png')
plt.close('all')


plt.tight_layout()
fig, axes = plt.subplots(nrows=z_fine, ncols=3)
for k in range(z_fine):
    ax = plt.subplot(1, z_fine, k + 1)
    im = plt.imshow(r2_co2_map[:, :, k, 0], origin='lower')
    plt.title("Layer" + str(k+1))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
fig.colorbar(im, cax=cbar_ax, shrink=0.50)
plt.savefig('r2_map_co2.png')
plt.close('all')


for nw in range(len(loc_x)):
    for t in range(1, 5, n_tsteps):
        plt.scatter(samples_f[:, nw, t]*max_sw, samples_c[:, nw, t]*(max_sw-min_sw) + min_sw, c='b', marker="s", label='coarse')
        plt.scatter(samples_f[:, nw, t]*max_sw, samples_r[:, nw, t]*(max_sw-min_sw) + min_sw, c='r', marker="o", label='rec')
        plt.xlim(0, np.max(samples_f[:, nw, t])*(max_sw-min_sw) + min_sw)
        plt.ylim(0, np.max(samples_f[:, nw, t])*(max_sw-min_sw) + min_sw)
        plt.legend(loc='upper left')
        plt.plot( [0, 1*(max_sw-min_sw) + min_sw], [0, 1*(max_sw-min_sw) + min_sw], linestyle='--', color='k' )
        plt.savefig('W' + str(nw+1) + '_t_' + str(t) + '_scatter.png')
        plt.close('all')

    for n in range(n_test_calc):
        plt.plot(time_steps, samples_r[n, nw, :] * (max_sw-min_sw) + min_sw, label="rec")
        plt.plot(time_steps, samples_f[n, nw, :] * (max_sw-min_sw) + min_sw, label="fine")
        plt.plot(time_steps, samples_c[n, nw, :] * (max_sw-min_sw) + min_sw, label="coarse")
        plt.legend(loc="upper left")
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Fraction', fontsize=12)
        plt.savefig('W' + str(nw + 1) + '_testsample_' + str(n) +'.png')
        plt.close('all')

    for n in range(n_test_calc):
        plt.plot(time_steps, samples_r_p[n, nw, :] * (max_p-min_p) + min_p, label="rec")
        plt.plot(time_steps, samples_f_p[n, nw, :] * (max_p-min_p) + min_p, label="fine")
        plt.plot(time_steps, samples_c_p[n, nw, :] * (max_p-min_p) + min_p, label="coarse")
        plt.legend(loc="upper left")
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Pressure (bars)', fontsize=12)
        plt.savefig('W' + str(nw + 1) + '_p_testsample_' + str(n) +'.png')
        plt.close('all')


    plt.plot(time_steps, np.mean(samples_r[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="rec")
    plt.plot(time_steps, np.mean(samples_f[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="fine")
    plt.plot(time_steps, np.mean(samples_c[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="coarse")
    plt.legend(loc="upper left")
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Fraction', fontsize=12)
    plt.savefig('W' + str(nw+1) + '_mean.png')
    plt.close('all')

    plt.plot(time_steps, np.std(samples_r[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="rec")
    plt.plot(time_steps, np.std(samples_f[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="fine")
    plt.plot(time_steps, np.std(samples_c[:, nw, :]*(max_sw-min_sw) + min_sw, axis=0), label="coarse")
    plt.legend(loc="upper left")
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Fraction', fontsize=12)
    plt.savefig('W' + str(nw+1) + '_std.png')
    plt.close('all')

    plt.plot(time_steps, np.mean(samples_r_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="rec")
    plt.plot(time_steps, np.mean(samples_f_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="fine")
    plt.plot(time_steps, np.mean(samples_c_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="coarse")
    plt.legend(loc="upper left")
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Pressure (bars)', fontsize=12)
    plt.savefig('W' + str(nw+1) + '_p_mean.png')
    plt.close('all')

    plt.plot(time_steps, np.std(samples_r_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="rec")
    plt.plot(time_steps, np.std(samples_f_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="fine")
    plt.plot(time_steps, np.std(samples_c_p[:, nw, :]*(max_p-min_p) + min_p, axis=0), label="coarse")
    plt.legend(loc="upper left")
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Pressure (bars)', fontsize=12)
    plt.savefig('W' + str(nw+1) + '_p_std.png')
    plt.close('all')


np.savetxt('R2test_p_avg.out', np.mean(r2_p, axis=1), delimiter=',')   # X is an array
np.savetxt('R2test_co2_avg.out', np.mean(r2_co2, axis=1), delimiter=',')   # X is an array
np.savetxt('MSEtest_p_avg.out', np.mean(e2_p, axis=1), delimiter=',')   # X is an array
np.savetxt('MSEtest_co2_avg.out', np.mean(e2_co2, axis=1), delimiter=',')   # X is an array

np.savetxt('R2test_p.out', r2_p, delimiter=',')   # X is an array
np.savetxt('R2test_co2.out', r2_co2, delimiter=',')   # X is an array
np.savetxt('MSEtest_p.out', e2_p, delimiter=',')   # X is an array
np.savetxt('MSEtest_co2.out', e2_co2, delimiter=',')   # X is an array

print(np.mean(r2_p))
print(np.mean(r2_co2))

print(np.mean(e2_p))
print(np.mean(e2_co2))

plt.plot(time_steps, np.mean(r2_p, axis=0))
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('R2', fontsize=12)
plt.savefig('R2_p_test_avg.png')
plt.close('all')

plt.plot(time_steps, np.mean(r2_co2, axis=0))
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('R2', fontsize=12)
plt.savefig('R2_co2_test_avg.png')
plt.close('all')


plt.plot(time_steps, np.mean(e2_p, axis=0))
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.savefig('e2_p_test_avg.png')
plt.close('all')

plt.plot(time_steps, np.mean(e2_co2, axis=0))
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.savefig('e2_co2_test_avg.png')
plt.close('all')


