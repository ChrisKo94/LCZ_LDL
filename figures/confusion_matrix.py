import numpy as np
import h5py
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys
import os
sys.path.insert(0, os.getcwd())
from utils import model

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

distributional = True

## Set path ##
path = os.getcwd()
path_data = Path(path, "data")

## Import data + limit to urban classes ##

test_data = test_data = h5py.File(Path(path_data, "testing.h5"),'r')
x_test = np.array(test_data.get("sen2"))
y_test = np.array(test_data.get("y"))

indices_test = np.where(np.where(y_test == np.amax(y_test, 0))[1] + 1 < 11)[0]
x_test = x_test[indices_test, :, :, :]
y_test = y_test[indices_test]

## Load model ##

model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10)

## Exemplary run (as shown in paper)
if distributional:
    res_ckpt_filepath = Path(path, "results", "Sen2LCZ_bs_64_lr_0.002_seed_1234_d_weights_best.hdf5")
else:
    res_ckpt_filepath = Path(path, "results", "Sen2LCZ_bs_64_lr_0.002_seed_1234_weights_best.hdf5")

model.load_weights(res_ckpt_filepath, by_name=False)

## Predict + save predictions ##

batchSize = setting_dict["Data"]["train_batch_size"]

y_pre_prob = model.predict(x_test, batch_size = batchSize)

true_labels = (np.argmax(y_test, axis=1) + 1)
pred_labels = (np.argmax(y_pre_prob, axis=1) + 1)
confidence = y_pre_prob[np.arange(y_pre_prob.shape[0]), (pred_labels - 1).tolist()]
prob_of_true_label = y_pre_prob[np.arange(y_pre_prob.shape[0]), (true_labels -1).tolist()]

entropies_results = pd.DataFrame({'true_class': true_labels, 'predicted_class': pred_labels,
                                  'confidence': confidence, 'probability_of_true_class': prob_of_true_label})

## Confusion Matrix ##

if distributional:
    conf_mat = confusion_matrix(entropies_results["true_class"],
                                entropies_results["predicted_class"], normalize="true")
else:
    conf_mat = confusion_matrix(entropies_results["label"],
                                entropies_results["prediction"], normalize="true")

labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
heat = sns.heatmap(conf_mat * 100, annot=True, fmt = ".0f", cmap="summer")
plt.xlabel('Predicted Class', fontsize = 15)
plt.ylabel('Majority Vote', fontsize = 15)