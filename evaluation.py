import numpy as np
import h5py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import json
import argparse
from pathlib import Path

from dataLoader import generator
import model
import lr

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
#import tensorflow_probability as tfp
#from tfp.stats import expected_calibration_error

#path = os.getcwd()
path = "D:/Data/LCZ_Votes/"

## Def data preparation ##

def predata4LCZ(file, keyX, keyY):
    hf = h5py.File(file, 'r')
    x_tra = np.array(hf[keyX])
    y_tra = np.array(hf[keyY])
    hf.close()

    return x_tra, y_tra

## Load test data ##

file = Path(path, "test_data.h5")

x_tst, y_tst= predata4LCZ(file, 'x', 'y')

test_label_distributions_h5 = h5py.File(Path(path,"test_label_distributions_data.h5"), "r")
test_label_distributions = np.array(test_label_distributions_h5.get("test_label_distributions"))

## Subset to urban classes (1-10) ##

indices_test = np.where(np.where(y_tst == np.amax(y_tst, 0))[1] + 1 < 11)[0]
x_tst = x_tst[indices_test, :, :, :]
y_tst = y_tst[indices_test, :10]

## Model prediction ##

def evaluation(res_ckpt_filepath):

    ## Model settings
    import model
    model = model.sen2LCZ_drop(depth=17,
                               dropRate=setting_dict["Data"]["dropout"],
                               fusion=setting_dict["Data"]["fusion"],
                               num_classes=setting_dict["Data"]["num_classes"])
    print("Model configured")

    model.load_weights(res_ckpt_filepath, by_name=False)
    y_pre_prob = model.predict(x_tst, batch_size = setting_dict["Data"]["test_batch_size"])
    y_pre = y_pre_prob.argmax(axis=-1)+1
    y_testV = y_tst.argmax(axis=-1)+1

    classRep = classification_report(y_testV, y_pre, digits=4, output_dict=True)
    oa = accuracy_score(y_testV, y_pre)
    macro_avg = classRep["macro avg"]["precision"]
    weighted_avg = classRep["weighted avg"]["precision"]
    cohKappa = cohen_kappa_score(y_testV, y_pre)

    cce = tf.keras.losses.CategoricalCrossentropy()

    ce_distr = float(cce(test_label_distributions, y_pre_prob).cpu().numpy())
    ce_one_hot = float(cce(y_tst, y_pre_prob).cpu().numpy())



    res = {
        'oa': float(oa),
        'maa': macro_avg,
        'waa': weighted_avg,
        'kappa': float(cohKappa),
        'ce_one_hot': ce_one_hot,
        'ce_distr': ce_distr
    }

    output_path_res = Path(res_ckpt_filepath.parent, f"{res_ckpt_filepath.stem}_results.json")
    output_path_res.parent.mkdir(parents=True, exist_ok=True)

    print(res)

    with open(output_path_res, 'w') as f:
        json.dump(res, f)
        print(res)

## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

## Train models ##

for seed in range(5):
    for distributional in [False, True]:
        for label_smoothing in [True, False]:
            setting_dict["Seed"] = seed

            seed = setting_dict["Seed"]
            label_smoothing = setting_dict["Calibration"]['label_smoothing']
            smoothing_param = setting_dict["Calibration"]['smoothing_param']
            batchSize = setting_dict["Data"]["train_batch_size"]
            lrate = setting_dict["Optimization"]["lr"]

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

            evaluation(res_ckpt_filepath)



