import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

## Set path ##
path = os.getcwd()

## Data generation ##

distr_urban_h5 = h5py.File(Path(path, "data", "label_distr_urban.h5"), 'r')
distr_urban = np.array(distr_urban_h5.get("label_distr_urban"))

entropies_urban = entropy(distr_urban, axis=1)

distr_nonurban_h5 = h5py.File(Path(path, "data", "label_distr_nonurban.h5"), 'r')
distr_nonurban = np.array(distr_nonurban_h5.get("label_distr_nonurban"))

entropies_nonurban = entropy(distr_nonurban, axis=1)

urban = pd.cut(entropies_urban,
               bins=[-0.1,0.2, 0.5, 0.7,1,1.3,np.inf],
               labels = ('0','0.3','0.6','0.8','1','1.2')).value_counts()

urban = urban / np.sum(urban)

urban = pd.DataFrame({'Entropy of Expert Votes Vector':['0','0.3','0.6','0.8','1','1.2'],
                      'Fraction':urban})

nonurban = pd.cut(entropies_nonurban,
                  bins=[-0.1,0.2, 0.5, 0.7,1,1.3,np.inf],
                  labels = ('0','0.3','0.6','0.8','1','1.2')).value_counts()

nonurban = nonurban / np.sum(nonurban)

nonurban = pd.DataFrame({'Entropy of Expert Votes Vector':['0','0.3','0.6','0.8','1','1.2'],
                         'Fraction':nonurban})

## Entropy Barplot ##

urban['Section']='Urban Classes'
nonurban['Section']='Non-Urban Classes'
res=pd.concat([nonurban,urban])

sns.barplot(x='Entropy of Expert Votes Vector',
            y='Fraction',
            data=res,
            hue='Section',
            palette='Set2')
plt.rcParams.update({'font.size': 20})