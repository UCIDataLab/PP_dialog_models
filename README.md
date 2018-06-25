# PP_dialog_models
Computational methods for evaluating patient-provider communication. <br>
Logistic regression and HMM can be used for prediction.

Author: Jihyun Park <jihyunp@uci.edu> <br>
Last updated: 6/22/2018

### load_model_and_predict.ipynb
Demo.
  iPython notebook file that loads the pre-trained model and the sample test data
  and predicts on the sample test data set.
  
### Input training and test file format 
Training and test file should have at least `visitid`, `talkturn`, 
`text`, `topicnumber`, `topicletter` as column names. <br>
For the test data, `topicnumber` and `topicletter` columns are not necessary 
since the test data can be run without labels. 
However, the scores will not be calculated without those columns.
  
- **sample_test_data.txt** <br>
A sample test data.
  
  
### models.py
Classes for models are in the file.
Details of the usage can be found in the demo iPython notebook file and the code docstring.

- `DialogModel` <br>
Base class for dialog model. Used when you have a set of results from another 
base model (independent model) that is trained somewhere else (e.g. output from RNN). 
Predictions and output probabilities are loaded using `load_model()` in this class object
 and then the object can be plugged into HMM. 

- `LogRegDialogModel` <br>
Class for running independent logistic regression model. <br>
`fit_model(tr_data)` to train data, `predict(te_data)` to make prediction.<br>

- `HMMDialogModel` <br>
Class for running Hidden Markov Model on top of some base independent model. <br>
`fit_model(tr_data)` to train data, `predict_viterbi(te_data)` to make prediction.<br>
 
- `DialogResult` <br>
Class that stores the results and calculates and prints out the scores.


### mhddata.py
Classes for the data. The classes loads the data and pre-processes. 
- `DialogData` : Base class for dialog data.
- `MHDTrainData` : Class for MHD training data. 
- `MHDTestData` : Class for MHD test data. 
Preprocessing methods are in **`preprocess.py`** file.
 
 
### hmm.py
Methods that are related to HMM


### utils.py 
Utility methods.


### ***.pkl Files
  Pre-trained models, vocabulary, and labels.