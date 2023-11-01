# Workflow of VeriNeuron Project
Welcome to this project, to follow or reproduce this work, please follow this ``workflow.md`` file.  
First, you need to prepare the environment: ``Python3.6 + Tensorflow2.0``  
Second, you need to prepare following packages: ``numpy,pandas,jobliob,scikit-learn``

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
- Third, we need to save the models after training.

```shell
# Please make sure you are at path: /training
# Then run the follow command:

python ./train_bank.py
python ./train_census.py
python ./train_credit.py
```

## 3. Global generation
```
```

## 4. Local generation
```
```

## 5. Retraining
```
```

## 6. Model evaluation
```
```

## 7. Comparison experiment with baseline
```
```