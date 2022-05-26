import h5py
import yaml
import json
import pandas as pd
from pathlib import Path
import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from utils.reliability_diagram import *

import tensorflow as tf

path = os.getcwd()
results_dir = Path(path, 'results')
results_dir.mkdir(parents=True, exist_ok=True)

## Import data + limit to urban classes ##

test_data = h5py.File(Path(path,"data","testing.h5"),'r')
x_test = np.array(test_data.get("sen2"))
y_test = np.array(test_data.get("y"))

indices_test = np.where(np.where(y_test == np.amax(y_test, 0))[1] + 1 < 11)[0]
x_test = x_test[indices_test, :, :, :]
y_test = y_test[indices_test, :10]

test_label_distributions = np.array(test_data.get("y_distributional_urban"))

## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

## Save results to dataframe
results = pd.DataFrame()

def evaluation(res_ckpt_filepath):

    from utils import model_mc_dropout
    model = model_mc_dropout.sen2LCZ_drop(depth=17,
                                          dropRate=setting_dict["Data"]["dropout"],
                                          fusion=setting_dict["Data"]["fusion"],
                                          num_classes=setting_dict["Data"]["num_classes"])

    model.load_weights(res_ckpt_filepath, by_name=False)

    # MC forward passes
    n_iter = 20

    mc_preds = np.zeros((len(test_label_distributions),10))
    for i in tqdm.tqdm(range(n_iter)):
        y_pred = model.predict(x_test, batch_size = setting_dict["Data"]["test_batch_size"])
        mc_preds = tf.add(mc_preds, y_pred)

    y_pre_prob = mc_preds.numpy() / n_iter

    true_labels = (np.argmax(y_test, axis=1) + 1)
    pred_labels = (np.argmax(y_pre_prob, axis=1) + 1)
    confidence = y_pre_prob[np.arange(y_pre_prob.shape[0]), (pred_labels - 1).tolist()]

    # Compute performance metrics
    classRep = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    oa = accuracy_score(true_labels, pred_labels)
    macro_avg = classRep["macro avg"]["precision"]
    weighted_avg = classRep["weighted avg"]["precision"]
    cohKappa = cohen_kappa_score(true_labels, pred_labels)
    # Derive cross-entropies and ece
    cce = tf.keras.losses.CategoricalCrossentropy()
    ce_distr = float(cce(test_label_distributions, y_pre_prob).cpu().numpy())
    ce_one_hot = float(cce(y_test, y_pre_prob).cpu().numpy())
    ece = compute_calibration(true_labels,pred_labels,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['expected_calibration_error']
    mce = compute_calibration(true_labels,pred_labels,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['max_calibration_error']
    sce = compute_calibration(true_labels,pred_labels,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['static_calibration_error']

    # Store results
    res = {
        'oa': float(oa),
        'maa': macro_avg,
        'waa': weighted_avg,
        'kappa': float(cohKappa),
        'ce_one_hot': ce_one_hot,
        'ce_distr': ce_distr,
        'ece': ece,
        'mce': mce,
        'sce': sce
    }
    # Create results file
    output_path_res = Path(os.getcwd(), "results", f"{res_ckpt_filepath.stem}_results.json")
    output_path_res.parent.mkdir(parents=True, exist_ok=True)
    # Write results to disk
    with open(output_path_res, 'w') as f:
        json.dump(res, f)

    return res

for label_smoothing in [False]:
    for seed in range(5):
        # Set hyperparameters accordingly
        setting_dict["Seed"] = seed
        smoothing_param = setting_dict["Calibration"]['smoothing_param']
        setting_dict["Calibration"]['label_smoothing'] = label_smoothing
        batchSize = setting_dict["Data"]["train_batch_size"]
        lrate = setting_dict["Optimization"]["lr"]
        if label_smoothing:
            res_ckpt_filepath = Path(path, "results",
                                     f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_ls_{smoothing_param}_weights_best.hdf5")
        else:
            res_ckpt_filepath = Path(path, "results",
                                     f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best.hdf5")
        # Start evaluation
        res = evaluation(res_ckpt_filepath)
        # Store results in overall results matrix
        results = results.append(res, ignore_index=True)

# Write ALL results to disk
results.to_csv(Path(path,"results","0.002_results_mc.csv"))

