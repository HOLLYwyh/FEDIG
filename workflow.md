# Workflow of FEDIG Project
Welcome to this project, to follow or reproduce this work, please follow this ``workflow.md`` file.  
First, you need to prepare the environment: ``Python3.6 + Tensorflow2.0``  
Second, you need to prepare following packages: ``numpy,pandas,jobliob,scikit-learn``

**Notice: All commands written in this document are on the Windows 11 OS. Commands may vary for different OS.
Please verify and make appropriate modifications accordingly.**

## 1. Preprocess Dataset
```
In this part, we need to prepare and preprocess the dataset 
so that we can train the neural network model.
``` 

- First, we need to download the dataset from the internet. You can get the URL of datasets from the comments in the files in the folder ``/preprocessing``.
- Second, we need to save the datasets into the folder ``/datasets/raw``.
- Third, we need to run the following **python command** to preprocess the data and save them in folder ``/datasets``.

```shell
# Please make sure you are at path: /preprocessing 
# Then run the follow command:

python ./pre_bank.py
python ./pre_census.py
python ./pre_credit.py
```

## 2. Model training
```
In this part, we need to spilt data into training dataset and test dataset.
Then we use the dataset to train three DNN models.
```
- First, we need to split the data we preprocessed into training dataset and test dataset.
- Second, we need to set the corresponding hyperparameters and train the neural network.
- Third, we need to save the models after training. Also, we need to evaluate the model.

```shell
# Please make sure you are at path: /training
# Then run the follow command:

python ./train_bank.py
python ./train_census.py
python ./train_credit.py
```

## 3. Cluster
```
Before generation, we need to do some preparatory work.
```
- In this part, we use K-Means algorithm to divide the dataset into clusters.

```shell
# Please make sure you are at path: /clusters
# The run the follow command:

python ./cluster.py
```

## 4. Comparison experiments with baseline
```
Some experiments of our algorithm and baselines.
```
In this part, we do some experiments of our algorithm FEDIG and baselines, 
which can prove that our algorithm is actually better than them.

**Notice:**  
1. **All the results of experiments are stored in folder */experiments/logfile*
 , so please clear all the *.csv* files before running the experiments.**
2. **Each experiment just run one time, in the paper we actually run five times and
calculate the average result of them. So if you want to get an accurate result, please run five times (or more) per experiment.**

-  First, we need to conduct some experiments to determine the parameters of our algorithm.
```shell
# Please make sure that you are at path : /experiments
# 1.1 eta (η) of FEDIG
python .\parameter\eta.py

# 1.2 min_len of FEDIG
python .\parameter\min_len.py
```
-  Second, we compare our algorithm FEDIG with baselines in four aspects.  
We use four RQs to evaluate our algorithm, every RQ has some related experiments.
    - RQ1: How effective is FEDIG in generating individual discriminatory instance?
    - RQ2: How efficient is FEDIG in generating individual discriminatory instances?
    - RQ3: How to explain the unfairness of DNNs with biased features identified by FEDIG?
    - RQ4: How useful are the generated test instances for improving the fairness of the DNN model?

1. RQ1: How effective is FEDIG in generating individual discriminatory instance?  
```shell
# We calculate GSR, time cost of four baselines and FEDIG on three datasets
# Plw
```

2. RQ2: How efficient is FEDIG in generating individual discriminatory instances?  
```shell

```
3. RQ3: How to explain the unfairness of DNNs with biased features identified by FEDIG?  
```shell

```
4. RQ4: How useful are the generated test instances for improving the fairness of the DNN model?  
```shell

```


## 6. Algorithm Evaluation
```
```
