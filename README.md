# drug_toxicity_predction
This project aims to enable users to predict the toxicity of drugs using the Deep-learning Multi-Layer Perceptron (MLP) algorithm, which has been meticulously implemented using PyTorch.
## Description
Cell and animal experiments have traditionally been used to determine drug toxicity, but they can be time-consuming, expensive, and labor-intensive. However, we now employ the deep learning MLP algorithm to create models based on known drug features and labels. By identifying the best model that predicts toxicity using the highest f1-score from the validation dataset, we can streamline the screening process. To facilitate the utilization of this model for drug toxicity prediction, a user-friendly software has been developed. This software only requires the input of the absolute path of the CSV file containing the drug features in order to generate accurate prediction results.
In this repository, we utilize the train_valid.py script to train a model. The data is located in the dataset folder and consists of two files: train_feature.pkl and hERG.pkl. These files contain 1974 samples with 729 features. We split this dataset into a training dataset and a validation dataset in a 9:1 ratio. Additionally, we employ the test.py script to obtain drug labels for a series of tests.

## Environment
* Python 3.9.13
* PyTorch 2.0.1
* Numpy 1.24.3
* Pandas 1.5.3
* Matplotlib 3.7.1
* Sklearn 1.2.2
* Argparse 1.1
## Neural Network Diagram
![Neural Network Diagram](https://github.com/anxuanhan/drug_toxicity_prediction/blob/main/img/neural%20network%20diagram.png)

## Train_Valid_Loss and Valid_F1_Score
![Train_Valid_Loss and Valid_F1_Score](https://github.com/anxuanhan/drug_toxicity_prediction/blob/main/img/loss%20and%20f1_score.jpg)

## Usage
Open cmd<br>
python software.py "the absolute path of the csv file"<br>
Users can run the script as follow, here is an example command:<br>
```
$ python software.py "D:\deeplearning_project\dataset.csv"
```
Then the predicted results will be shown in the cmd line.<br>

Note: Here are the codes for training that can be used to reproduce the results and train your own data:
```
$ python train_valid.py
```

## Dataset
The dataset is derived from Huawei Cup 2021 mathematical modeling problem D<br>
[Data source](https://cpipc.acge.org.cn//cw/detail/4/2c9080147c73b890017c7779e57e07d2)


## Contact
Any questions, problems, bugs are welcome and should be dumped to Anxuan Han. hax3417@163.com.<br>
Created on July. 24, 2023.  
