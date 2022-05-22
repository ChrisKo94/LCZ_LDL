import os
import h5py
import yaml
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.activations import softmax
from utils.calibration import compute_calibration

## Set path ##
path = os.getcwd()

## Import data ##

validation_file = Path(path,"data", "validation_data.h5")
validation_data = h5py.File(validation_file, 'r')
x_val = np.array(validation_data.get("sen2"))
y_val = np.array(validation_data.get("y"))

## Subset to urban classes (1-10) ##

indices_val = np.where(np.where(y_val == np.amax(y_val, 0))[1] + 1 < 11)[0]
x_val = x_val[indices_val, :, :, :]
y_val = y_val[indices_val,:10]

## Load label distributions

y_val_d = np.array(validation_data.get('y_distributional_urban'))

## Load test data ##

test_file = Path(path, "data", "test_data.h5")
test_data = h5py.File(test_file, 'r')
x_test = np.array(test_data.get("sen2"))
y_test = np.array(test_data.get("y"))

y_test_d = np.array(test_data.get("y_distributional_urban"))

## Subset to urban classes (1-10) ##

indices_test = np.where(np.where(y_test == np.amax(y_test, 0))[1] + 1 < 11)[0]
x_test = x_test[indices_test, :, :, :]
y_test = y_test[indices_test, :10]

## Save results to dataframe
results = pd.DataFrame()

def temp_scaler(res_ckpt_filepath):
    from utils import model_softmax
    ## Model settings, note: Using model that predicts logits instead of probabilites
    model = model_softmax.sen2LCZ_drop(depth=17,
                                       dropRate=setting_dict["Data"]["dropout"],
                                       fusion=setting_dict["Data"]["fusion"],
                                       num_classes=setting_dict["Data"]["num_classes"])
    print("Model configured")

    model.load_weights(res_ckpt_filepath, by_name=False)
    y_pred = model.predict(x_val, batch_size = setting_dict["Data"]["test_batch_size"])

    if distributional:
        y_val_actual = y_val_d
    else:
        y_val_actual = y_val
    # Define NLL based on logits
    def compute_loss():
        y_pred_model_w_temp = tf.math.divide(y_pred, temp)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(tf.convert_to_tensor(y_val_actual),
                                                    y_pred_model_w_temp))
        return loss
    # Define initial trainable temperature variable
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    # Configure optimizer for minimizing NLL
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    print('Temperature Initial value: {}'.format(temp.numpy()))
    # Optimize NLL with max. 10k steps
    for i in range(10000):
         opts = optimizer.minimize(compute_loss, var_list=[temp])

    print('Temperature Final value: {}'.format(temp.numpy()))

    # Predict on Test set
    y_pred = model.predict(x_test, batch_size=setting_dict["Data"]["test_batch_size"])
    # Derive temperature scaled logits
    y_pred_model_w_temp = tf.math.divide(y_pred, temp)
    # Softmax transformation of scaled logits
    y_pred_prob = softmax(y_pred_model_w_temp).numpy()
    # Derive predictions + corr. confidence
    y_pre = y_pred_prob.argmax(axis=-1) + 1
    confidence = y_pred_prob[np.arange(y_pred_prob.shape[0]), (y_pre - 1).tolist()]
    y_testV = y_test.argmax(axis=-1) + 1
    # Compute cross-entropies and ece
    ce_one_hot = float(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(tf.convert_to_tensor(y_test),
                                                y_pred_model_w_temp)).cpu().numpy())
    ce_distr = float(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(tf.convert_to_tensor(y_test_d),
                                                y_pred_model_w_temp)).cpu().numpy())

    ece = compute_calibration(y_testV, y_pre, confidence, y_pred_prob, num_bins=25)['expected_calibration_error']
    mce = compute_calibration(y_testV, y_pre, confidence, y_pred_prob, num_bins=25)['max_calibration_error']
    sce = compute_calibration(y_testV, y_pre, confidence, y_pred_prob, num_bins=25)['static_calibration_error']

    # Store results
    res = {
        'ce_one_hot': ce_one_hot,
        'ce_distr': ce_distr,
        'ece': ece,
        'mce': mce,
        'sce': sce
    }
    # Create results file
    output_path_res = Path(res_ckpt_filepath.parent, f"{res_ckpt_filepath.stem}_results_ts.json")
    output_path_res.parent.mkdir(parents=True, exist_ok=True)
    # Save results
    with open(output_path_res, 'w') as f:
        json.dump(res, f)
        print("Starting Evaluating: Distributional = " +
              str(distributional) + ", label smoothing = " +
              str(label_smoothing))
        print(res)

    return res

## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

## Train models ##

for distributional in [False, True]:
    for label_smoothing in [True, False]:
        for seed in range(5):
            # Set hyperparameters accordingly
            setting_dict["Seed"] = seed
            setting_dict["Calibration"]['label_smoothing'] = label_smoothing
            setting_dict["Data"]["distributional"] = distributional
            smoothing_param = setting_dict["Calibration"]['smoothing_param']
            batchSize = setting_dict["Data"]["train_batch_size"]
            lrate = setting_dict["Optimization"]["lr"]
            # Stitch together model checkpoint filename
            if distributional:
                if label_smoothing:
                    res_ckpt_filepath = Path(path, "results",
                                             f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_d_ls_{smoothing_param}_weights_best.hdf5")
                else:
                    res_ckpt_filepath = Path(path, "results",
                                     f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_d_weights_best.hdf5")
            else:
                if label_smoothing:
                    res_ckpt_filepath = Path(path, "results",
                                     f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_ls_{smoothing_param}_weights_best.hdf5")
                else:
                    res_ckpt_filepath = Path(path, "results",
                                     f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best.hdf5")
            # Start temperature scaling
            res = temp_scaler(res_ckpt_filepath)
            # Append results to final results matrix
            results = results.append(res, ignore_index=True)
# Save ALL results to disk
results.to_csv(Path(path,"results","0.002_results_ts.csv"))