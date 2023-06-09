import os
import h5py
import argparse
import numpy as np
import yaml
import random as rn
from numpy import random
from pathlib import Path

from utils.dataLoader import generator
from utils import lr
from utils.model import *

import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

## Single GPU case ##
gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

## Set path ##
path = os.getcwd()
results_dir = Path(path, 'results')
results_dir.mkdir(parents=True, exist_ok=True)

## Import data ##
train_file = Path(path,"data", "training.h5")
train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("sen2"))
y_train = np.array(train_data.get("y"))

validation_file = Path(path,"data", "validation.h5")
validation_data = h5py.File(validation_file, 'r')
x_val = np.array(validation_data.get("sen2"))
y_val = np.array(validation_data.get("y"))

## Subset to urban classes (1-10) ##

indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
x_train = x_train[indices_train, :, :, :]
y_train = y_train[indices_train]

indices_val = np.where(np.where(y_val == np.amax(y_val, 0))[1] + 1 < 11)[0]
x_val = x_val[indices_val, :, :, :]
y_val = y_val[indices_val]

## Load label distributions

y_train_d = np.array(train_data.get('y_distributional_urban'))

y_val_d = np.array(validation_data.get('y_distributional_urban'))


## Def model training ##

def train_model(setting_dict: dict):
    seed = setting_dict["Seed"]
    label_smoothing = setting_dict["Calibration"]['label_smoothing']
    smoothing_param = setting_dict["Calibration"]['smoothing_param']
    path = os.getcwd()

    ## Reproducibility
    random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = '0'

    ## Model settings
    model = sen2LCZ_drop(depth=17,
                               dropRate=setting_dict["Data"]["dropout"],
                               fusion=setting_dict["Data"]["fusion"],
                               num_classes=setting_dict["Data"]["num_classes"])
    print("Model configured")

    if distributional:
        model.compile(optimizer=Nadam(),
                      loss='KLDivergence',
                      metrics=['KLDivergence'])
    else:
        model.compile(optimizer=Nadam(),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
    print("Model compiled")

    trainNumber = y_train.shape[0]
    validationNumber = y_val.shape[0]
    batchSize = setting_dict["Data"]["train_batch_size"]
    lrate = setting_dict["Optimization"]["lr"]

    lr_sched = lr.step_decay_schedule(initial_lr=lrate,
                                      decay_factor=0.5,
                                      step_size=5)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=setting_dict["Optimization"]["patience"])

    if distributional:
        if label_smoothing:
            ckpt_file = Path(
                path, "results",
                f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_d_ls_{smoothing_param}_weights_best.hdf5")
            y_train_actual = (1 - smoothing_param) * y_train_d + (smoothing_param / y_train_d.shape[1])
            y_val_actual = y_val_d
        else:
            ckpt_file = Path(
                path, "results",
                f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_d_weights_best.hdf5")
            y_train_actual = y_train_d
            y_val_actual = y_val_d
        checkpoint = ModelCheckpoint(
            ckpt_file,
            monitor='val_kullback_leibler_divergence',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch')
    else:
        if label_smoothing:
            ckpt_file = Path(path, "results",
                             f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_ls_{smoothing_param}_weights_best.hdf5")
            y_train_actual = (1 - smoothing_param) * y_train + (smoothing_param / y_train.shape[1])
            y_val_actual = y_val
        else:
            ckpt_file = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best.hdf5")
            y_train_actual = y_train
            y_val_actual = y_val
        checkpoint = ModelCheckpoint(
            ckpt_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch')


    print("Callbacks and checkpoint initialized")
    print("Starting Training: Distributional = " + str(distributional) + ", label smoothing = " + str(label_smoothing))
    model.fit(generator(x_train,
                        y_train_actual,
                        batchSize=batchSize,
                        num=trainNumber,
                        mode="urban"),
                        steps_per_epoch=trainNumber // batchSize,
                        validation_data=generator(x_val,
                                                  y_val_actual,
                                                  num=validationNumber,
                                                  batchSize=batchSize,
                                                  mode="urban"),
                        validation_steps=validationNumber // batchSize,
                        epochs=setting_dict["Trainer"]["max_epochs"],
                        max_queue_size=100,
                        callbacks=[early_stopping, checkpoint, lr_sched])


## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--single_run', type=bool, required=False, default = False)
args = parser.parse_args()

## Train models ##

if __name__ == "__main__":
    if args.single_run:
        smoothing_param = setting_dict["Calibration"]['smoothing_param']
        distributional = setting_dict["Data"]["distributional"]
        train_model(setting_dict)
    else:
        for distributional in [False, True]:
            for label_smoothing in [True, False]:
                for seed in range(5):
                    setting_dict["Seed"] = seed
                    setting_dict["Calibration"]['label_smoothing'] = label_smoothing
                    setting_dict["Data"]["distributional"] = distributional
                    smoothing_param = setting_dict["Calibration"]['smoothing_param']
                    train_model(setting_dict)


