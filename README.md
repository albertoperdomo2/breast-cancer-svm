# SVM script for breast cancer detection

## Prerequisites: 

```
Python 3.5 - 3.8
Libraries: SciKit-Learn, Numpy, Pandas, Matplotlib and Seaborn. 
```
 
Cancer is one of the main causes of death along the entire world and reports of the World Health Organization state that almost the 30-50% of the new diagnosed cases each year could be pre- vented. Thanks to Artificial Intelligence and machine learning, nowadays there are more support tools for doctors that can help to early detect and diagnose this type of disease, because here as in many other fields, times really matter. So, the purpose of this project is to explain the development of a first version of a support tool for early detect breast cancer, predicting the nature (malign or benign) of the studied breast mass.

The dataset can be downloaded from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data since it is publicly available. 

In the "Others" folder, there can be found a paper regarding this project and other testing scripts which were intended to extract the data from FNA images, considering that the mentioned dataset already has the data "digitized". 

Support Vector Machines is a highly convenient model when working in high dimensional spaces and it is proved that it reports a great effectiveness, efficienty and accuracy. In this case, it was selected due to different factors: SVM presents the possibility to use a custom kernel, which is an algorithm which recognizes patterns in the data and can make the model more suitable for the current dataset. Also, the SVM is really recommended for two class datasets, which is the case of the data in this project, having only benign or malign cancer; and this is not as common as it may seem at a first sight, because usually medical tools contemplate the possibility of having a healthy patient, so then there is no binary classification because there are three classes.

The script itself has all the comments made when writting it, so I think it is self explanatory. In case that a deeper explanation is needed, the paper in "Others" folder can help to understand better what is being done. 
