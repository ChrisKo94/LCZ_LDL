import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Note: voting dataset will be added in a future release
voting = pd.DataFrame(y_test, columns=["vote", "orig_label"])

conf_mat = confusion_matrix(voting["orig_label"], voting["vote"], normalize="true")
labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "B", "C", "D", "E", "F", "G"]

conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)

sns.heatmap(conf_mat.iloc[0:10,0:10] * 100, annot=True, fmt = ".0f", cmap="summer", cbar = False)
plt.show()