# Local Climate Zones Label Distribution Learning (LCZ_LDL)

---

The experiments are based on the So2Sat LCZ42 data set [1], 
which includes aerial satellite images of cities around the world from the Sentinel satellite mission. 
Every satellite image is linked to either a single Local Climate Zone (LCZ) class (original data set), or has a 
corresponding vector of label votes from multiple human annotators (future release). 
The original data set can be downloaded at https://mediatum.ub.tum.de/1454690 (and referenced pages therein). 
In our work we study a particular subset of the So2Sat LCZ42 data set, which yields information about the labeling 
process and has not yet been published. This is however planned for the near future and will be most likely published
via the same stream as the original data set. 
The individual python executable files are structured as follows: 

- train.py: Training script, saves results to 'results' folder
- evaluation.py: Evaluation script, computes all metrics shown in paper and saves results to 'results' folder
- temp_scaling.py: Performs temperature scaling on trained model(s), saves results to 'results' folder
- configs: 
  - model_settings.yaml: Stores model settings if single run is desired
- utils:
  - model.py: Sen2LCZ model as presented in [2]. Corresponding repository: 
https://github.com/ChunpingQiu/benchmark-on-So2SatLCZ42-dataset-a-simple-tour
  - model_softmax.py: Adapted version of Sen2LCZ which returns logits instead of probabilities
  - reliability_diagram.py: Code for computing calibration metrics and for visualizing model calibration 
via reliability table, adapted from https://github.com/hollance/reliability-diagrams
  - caliibration.py: Sub-function of reliability_diagram.py 
  - dataLoader.py: Custom data loader
  - lr.py: Custom learning rate scheduler
- figures:
  - confusion_matrix: Diplays confusion matrices as shown in paper
  - entropy_barplot: Grouped barplot of voting entropies
  - reliability_diagram: Diplays reliability diagrams as shown in paper
  - voting_confusion: Confusion matrix between majority vote and individual vote
- results: Stores results in the form of model weights and model performance metrics evaluated on test set 
- data: Stores data sets, in particular train/val/test, voting counts, entropies, label distributions, ... 
(will be released in the near future by the original authors of [1])

## Requirements

---

`pip install -U -r requirements.txt`

## Training

---

All configs: `python train.py `

## Testing

---

All configs: `python evaluation.py`

## Temperature Scaling

All configs: `python temp_scaling.py `

---

## Figures

All scripts are python executables and compute the images as seen in the paper. 

---



### References

[1]: Zhu, X. X., Hu, J., Qiu, C., Shi, Y., Kang, J., Mou, L., ... & Wang, Y. (2020). So2Sat LCZ42: 
a benchmark data set for the classification of global local climate zones [Software and Data Sets]. 
IEEE Geoscience and Remote Sensing Magazine, 8(3), 76-89.

[2]: Qiu, C., Tong, X., Schmitt, M., Bechtel, B., & Zhu, X. X. (2020). Multilevel feature fusion-based CNN
for local climate zone classification from sentinel-2 images: 
Benchmark results on the So2Sat LCZ42 dataset. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 2793-2806.
