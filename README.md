# drug_toxicity_predction
This project is for users to predict the toxicity of drugs, using the Multi-Layer Perceptron (MLP) algorithm, which is carefully implemented by pytorch.
## Description
Cell and animal experiences are the traditional way to determine whether a drug is toxic, but it can waste a lot of time, money and effort to screen it. Now we use the  deeplearning MLP algorithm to model according to the known drug features and drug labels, and find the best model to predict toxicity based on the maximum value of f1-score from the validation dataset. In order to help people better use the model to predict drug toxicity, a minor software was written, which only needs to input the absolute path of the csv file that holds the drug characteristics to get the prediction results.
## Environment
* Python 3.9.13
* PyTorch 2.0.1
* Numpy 1.24.3
* Pandas 1.5.3
* Matplotlib 3.7.1
* Sklearn 1.2.2
* Argparse 1.1
* Pycharm 2023.1.2
## Neural Network Diagram
![Neural Network Diagram](https://github.com/anxuanhan/drug_toxicity_prediction/blob/main/img/neural%20network%20diagram.png)

## Usage
open cmd<br>
python software.py "the absolute path of the csv file"<br>
The following command is an example:<br>
```
$ python software.py "D:\deeplearning project\dataset.csv"
```
Then the predicted results will be returned in the cmd window.

## Dataset
The dataset is derived from Huawei Cup 2021 mathematical modeling problem D<br>
[Data source](https://cpipc.acge.org.cn//cw/detail/4/2c9080147c73b890017c7779e57e07d2)


## Contact
Any questions, problems, bugs are welcome and should be dumped to Anxuan Han. hax3417@163.com.<br>
Created on July. 24, 2023.  
