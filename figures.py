import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import model

from utils.reliability_diagram import *

## Import data + limit to urban classes ##

test_data = h5py.File(path_data + "test_data.h5",'r')
x_test = np.array(test_data.get("x"))
y_test = np.array(test_data.get("y"))

indices_test = np.where(np.where(y_test == np.amax(y_test, 0))[1] + 1 < 11)[0]
x_test = x_test[indices_test, :, :, :]
y_test = y_test[indices_test]

## Load model ##

model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10)

model.load_weights(res_ckpt_filepath, by_name=False)

## Predict + save predictions ##

y_pre_prob = model.predict(x_test, batch_size = batchSize)

true_labels = (np.argmax(y_test, axis=1) + 1)
pred_labels = (np.argmax(y_pre_prob, axis=1) + 1)
confidence = y_pre_prob[np.arange(y_pre_prob.shape[0]), (pred_labels - 1).tolist()]
prob_of_true_label = y_pre_prob[np.arange(y_pre_prob.shape[0]), (true_labels -1).tolist()]

test_mat = pd.DataFrame({'true_class': true_labels, 'predicted_class': pred_labels,
                         'confidence': confidence, 'probability_of_true_class': prob_of_true_label})


## Voting Confusion ##

## Confusion Matrix ##

if distributional:
    conf_mat = confusion_matrix(entropies_results["true_class"],
                                entropies_results["predicted_class"], normalize="true")
else:
    conf_mat = confusion_matrix(entropies_results["label"],
                                entropies_results["prediction"], normalize="true")

if urban:
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
else:
    labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "B", "C", "D", "E", "F", "G"]

conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
heat = sns.heatmap(conf_mat * 100, annot=True, fmt = ".0f", cmap="summer")
plt.xlabel('Predicted Class', fontsize = 15)
plt.ylabel('Majority Vote', fontsize = 15)

## Reliability Diagram ##

y_true = reliability_results.true_class
y_pred = reliability_results.predicted_class
y_conf = reliability_results.confidence

# Override matplotlib default styling.
plt.style.use("seaborn")

plt.rc("font", size=16)
plt.rc("axes", labelsize=16)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=16)

plt.rc("axes", titlesize=20)
plt.rc("figure", titlesize=20)

fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=20, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          figsize=(6, 6), dpi=100,
                          title="",
                          return_fig=True)