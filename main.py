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
    inputs/targets for network training, defines and trains two models
    (for saturation and pressure), and saves the trained networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from base_model import model_create_two_state, r_squared, ssim_loss
import mat73  # add this



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


# -----------------------------------------------------------------------------
# 8. Prepare training datasets and shuffle
# -----------------------------------------------------------------------------

x_train_p_a = X_p_n_re[0:n_train*n_tsteps, :, :, :, :]
x_train_co2_a = X_sW_n_re[0:n_train*n_tsteps, :, :, :, :]
x_train_perm = X_perm_t[0:n_train*n_tsteps, :, :, :, :]
x_train_co2_p = X_sW_n_p[0:n_train*n_tsteps, :, :, :, :]
x_train_p_p = X_p_n_p[0:n_train*n_tsteps, :, :, :, :]
y_train_p = Y_p_n[0:n_train*n_tsteps, :, :, :,:]
y_train_co2 = Y_sW_n[0:n_train*n_tsteps, :, :, :,:]
x_train_dco2 = X_dsW_n_re[0:n_train*n_tsteps, :, :, :,:]
x_train_dp = X_dp_n_re[0:n_train*n_tsteps, :, :, :,:]


shuffled_idx = np.random.permutation(n_train*n_tsteps)
y_train_p_shuff = np.zeros((n_train*n_tsteps, x_fine, y_fine, z_fine,1), dtype="float32")
y_train_co2_shuff = np.zeros((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")
x_train_p_a_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_co2_a_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_perm_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_co2_p_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_p_p_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_dco2_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1
x_train_dp_shuff = np.ones((n_train*n_tsteps, x_fine, y_fine, z_fine, 1), dtype="float32")*-1

for i in range(n_train*n_tsteps):
    y_train_p_shuff[shuffled_idx[i]] = y_train_p[i:i+1]
    y_train_co2_shuff[shuffled_idx[i]] = y_train_co2[i:i+1]
    x_train_p_a_shuff[shuffled_idx[i]] = x_train_p_a[i:i+1]
    x_train_co2_a_shuff[shuffled_idx[i]] = x_train_co2_a[i:i+1]
    x_train_perm_shuff[shuffled_idx[i]] = x_train_perm[i:i+1]
    x_train_co2_p_shuff[shuffled_idx[i]] = x_train_co2_p[i:i+1]
    x_train_p_p_shuff[shuffled_idx[i]] = x_train_p_p[i:i+1]
    x_train_dco2_shuff[shuffled_idx[i]] = x_train_dco2[i:i+1]
    x_train_dp_shuff[shuffled_idx[i]] = x_train_dp[i:i+1]



# -----------------------------------------------------------------------------
# 9. Training settings
# -----------------------------------------------------------------------------

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=8, min_lr=0.00001)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="check_point",
    save_best_only=True,
    monitor='val_loss',
    )

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-7,
    patience=20,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True)


# -----------------------------------------------------------------------------
# 10. Network initialization 
# -----------------------------------------------------------------------------

model_co2 = model_create_two_state(x_fine, y_fine, z_fine, x_coarse, y_coarse, z_coarse, ks_x, ks_y, ks_z)
print(model_co2.summary())


model_p = model_create_two_state(x_fine, y_fine, z_fine, x_coarse, y_coarse, z_coarse, ks_x, ks_y, ks_z)
print(model_p.summary())


print("===================================================================")
print(r_squared(y_train_co2_shuff, x_train_co2_a_shuff))
print(r_squared(y_train_p_shuff, x_train_p_a_shuff))


print("===================================================================")
print(ssim_loss(y_train_co2_shuff, x_train_co2_a_shuff))
print(ssim_loss(y_train_p_shuff, x_train_p_a_shuff))
print("===================================================================")


# -----------------------------------------------------------------------------
# 11. Network training 
# -----------------------------------------------------------------------------

history_p = model_p.fit(
    [x_train_perm_shuff,
        x_train_p_a_shuff,
        x_train_co2_a_shuff,
        x_train_p_p_shuff,
        x_train_co2_p_shuff,
        x_train_dp_shuff],
    y_train_p_shuff,
    epochs=n_epochs, batch_size=batch_size, validation_split=0.1, validation_freq=1, use_multiprocessing=True,
    workers=4,
    # callbacks=[callback, reduce_lr, model_checkpoint_callback]
    callbacks=[callback, reduce_lr])
model_p.save('model_p')


history_co2 = model_co2.fit(
[x_train_perm_shuff,
    x_train_co2_a_shuff,
    x_train_p_a_shuff,
    x_train_co2_p_shuff,
    x_train_p_p_shuff,
    x_train_dco2_shuff],
y_train_co2_shuff,
epochs=n_epochs, batch_size=batch_size, validation_split=0.1, validation_freq=1, use_multiprocessing=True,
workers=4,
# callbacks=[callback, reduce_lr, model_checkpoint_callback]
callbacks=[callback, reduce_lr])
model_co2.save('model_co2')


# __________________________ post processing ________________________________________
plt.figure(figsize=(20, 4))
plt.semilogy(np.square(history_p.history['root_mean_squared_error']))
plt.semilogy(np.square(history_p.history['val_root_mean_squared_error']), '--')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('mean_squared_error_semilogy_p.png')
plt.figure(figsize=(20, 4))
plt.semilogy(np.square(history_co2.history['root_mean_squared_error']))
plt.semilogy(np.square(history_co2.history['val_root_mean_squared_error']),'--')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('mean_squared_error_semilogy_co2.png')
