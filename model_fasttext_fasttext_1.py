from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import os, os.path
import numpy as np
import json
import re
import ast
from nltk.tokenize import word_tokenize
import json
import pandas as pd
from scipy.stats import randint
#import seaborn as sns # used for plot interactive graph. 
# https://stackoverflow.com/questions/3453188/matplotlib-display-plot-on-a-remote-machine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # should come after .use('somebackend')
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from langdetect import detect
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import ast
import re
import decimal
import pickle
from functools import reduce
from operator import or_
import unidecode
from sklearn.metrics import roc_curve, auc, roc_auc_score
import fasttext
from fasttext import load_model
import io
import time
from sklearn.model_selection import KFold
import multiprocessing

################# definitions
# The optimal cut-off would be where the true positive rate (tpr) is high
# and the false positive rate (fpr) is low,
# and tpr (- fpr) is zero or near to zero
def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model
        related to the event rate

    Parameters
    ----------
    target: Matrix with dependent or target data, where rows are observations
    predicted_ Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def from_bin_to_vec(embeddings, vec_path):
    """ Produce from embeddings .bin file to .vec file and save it in vec_path.
        The .vec file is used for pretrainedVectors parameter for training the fastext model
    
    Parameters
    ----------
    embeddings: .bin file of the embeddings
    vec_path: path where the produced .vec file will be stored

    Returns
    -------
    returns nothing, but produces a .vec file of the embeddings and saves it in vec_path

    """
    f = embeddings
    # get all words from model
    words = f.get_words()
    with open(vec_path,'w') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass

def get_prediction_scores(trained_ft_model,X_test):
    """ Returns probability predictions of being labeled as positive on a set X_test.
        Fits for Fasttext.

    Parameters
    ----------
    trained_ft_model: trained supervised fasttext model
    X_test: set being predicted 

    Returns
    -------
    list of probabilities for being labeled positive on set X_test

    """
    ## generate class probabilites, where:
       # 1st element is probability for negative class,
       # 2nd element gives probability for positive class
    probs_ft = []
    for i in range(0,len(X_test)):
        # fasttext predict arguments: string, k=int
        # Given a string, get a list of labels (k) and a list of corresponding probabilities
        result = trained_ft_model.predict(X_test[i], k=2) # k=2 because we have 2 labels
        # result = ('__label__0', '__label__1'), array([0.972..., 0.027...])) 
        # first element is always the one with the higher probabiliy.
        # We want the first element always belongs to no and second to yes 
        if result[0][0] == '__label__0':
            probs_ft.append(result[1].tolist())
        elif result[0][0] == '__label__1':
            yes = result[1].tolist()[0]
            no = result[1].tolist()[1]
            probs = []
            probs.append(no)
            probs.append(yes)
            probs_ft.append(probs)
    ## take probabilites for being labeled positive
    y_scores_ft = [el[1] for el in probs_ft]
    return y_scores_ft


def get_predictions(threshold, prediction_scores):
    """ Apply the threshold to the prediction probability scores
        and return the resulting label predictions

    Parameters
    ----------
    threshold: the threshold we want to apply
    prediction_scores: prediction probability scores, i.e. list of probabilites of being labeled positive

    Returns
    -------
    list of label predictions, i.e. list of 1's and 0's

    """
    predictions = []
    for score in prediction_scores:
        if score >= threshold[0]:
            predictions.append(1)
        elif score < threshold[0]:
            predictions.append(0)
    return predictions

def get_parameter_value_with_results(i, param, param_values, params_wordembeddings, params_training, tune, X_test, y_test):
    print(str(i))
    model_name = "test_" + param + "_" + str(i)
    # bin_path = "word_vectors/fasttext/" + model_name + ".bin" 
    vec_path = "word_vectors/fasttext/" + model_name + ".vec" 
    if tune == "wordembeddings": ####### tuning parameter for fasttext WORD EMBEDDING
        params_wordembeddings[param] = param_values[i]
    embeddings = fasttext.train_unsupervised(input='data.txt', model='skipgram', **params_wordembeddings) 
    # embeddings.save_model(bin_path)
    # embeddings = load_model(bin_path)
    ### convert from fasttext embeddings (would be saved as .bin) to .vec,
    #   in order to use the embeddings .vec file as pretrainedVectors for fasttext text classification
    from_bin_to_vec(embeddings, vec_path)
    if tune == "training": ####### tuning parameter for fasttext TRAINING
        params_training[param] = param_values[i]
    # dimension of embeddings has to fit with dimension of look-up table (embeddings) in training model
    params_training["dim"] = embeddings.get_dimension()
    trained_model = fasttext.train_supervised(input=train_file, pretrainedVectors= vec_path, **params_training)
    ### find and apply optimal (threshold) cutoff point
    # get scores, i.e. list of probabilities for being labeled positive on set X_test
    y_scores = get_prediction_scores(trained_model,X_test)
    # find optimal probability threshold
    opt_threshold = find_optimal_cutoff(y_test, y_scores)
    # apply optimal threshold to the prediction probability and get label predictions
    y_pred = get_predictions(opt_threshold, y_scores) 
    ################## Evaluation
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    auprc = metrics.average_precision_score(y_test, y_pred)
    return [accuracy, precision, recall, auc, auprc]


def get_best_parameter_value_with_results(param, param_values, params_wordembeddings, params_training, tune, score, X_test, y_test):
    """ Get the best parameter value of 'param' among 'param_values' along with its metric results,
        which tunes the wordembeddings or training model ('tune'),
        based on wordembeddings parameter and training parameter settings ('params_wordembeddings' and params_training resp.),
        on a test set ('X_test', 'y_test').
        The best parameter value is evaluated according to metric score 'score'

    Parameters
    ----------
    param: parameter name, whose values we want to tune (which value is the best one) (e.g. "dim")
    param_values: list of values of param we wan to tune (e.g. [100, 200, 300])
    params_wordembeddings: dict of param:value of the other wordembeddings parameters and their values
            that we want to set before fine-tuning the value for param. (e.g. ["minn":2, "maxn":5])
            (normally these param:value have been already fine-tuned before)
    params_training: dict of param:value of the other training parameters and their values
            that we want to set before fine-tuning the value for param. (e.g. ["epoch":10, "lr":0.2])
            (normally these param:value have been already fine-tuned before)
    tune: accepts either 'wordembeddings' or 'training', depending on what we want to tune.
    score: accepts following: 'auprc', 'auc', 'recall', 'precision', 'accuracy',
           depending on which evaluation metric we want to apply
    X_test: list of strings (e.g. ["some preprocessed text 0", "some preprocessed text 1", ...]), 
            which represents the features of the set we want to test.
    y_test: list of 0's and 1's (e.g. [0,0,1,0,1,1,...]), 
            which represents the labels of the set want to test.

    Returns
    -------
    the best value of 'param' among the values in 'param_values', according to metric score 'score'
    """
    i_list = list(range(len(param_values)))
    pool = multiprocessing.Pool(processes=len(param_values))
    f=partial(get_parameter_value_with_results, param=param, param_values=param_values, params_wordembeddings=params_wordembeddings, params_training = params_training, tune = tune, X_test = X_test, y_test = y_test) 
    list_of_results = pool.map(f, i_list) #returns list of lists   
    accuracy_scores = [item[0] for item in list_of_results]
    precision_scores = [item[1] for item in list_of_results]
    recall_scores = [item[2] for item in list_of_results]
    auc_cores = [item[3] for item in list_of_results] # auc scores
    auprc_scores = [item[4] for item in list_of_results] # auprc scores    
    if score == "auprc":
        index_max = max(range(len(auprc_scores)), key=auprc_scores.__getitem__)
    elif score == "auc":
        index_max = max(range(len(auc_scores)), key=auc_scores.__getitem__)
    elif score == "recall":
        index_max = max(range(len(recall_scores)), key=recall_scores.__getitem__)
    elif score == "precision":
        index_max = max(range(len(precision_scores)), key=precision_scores.__getitem__)
    elif score == "accuracy":
        index_max = max(range(len(accuracy_scores)), key=accuracy_scores.__getitem__)
    results = {}
    results["auprc"] = auprc_scores[index_max]
    results["auc"] = auc_scores[index_max]
    results["recall"] = recall_scores[index_max]
    results["precision"] = precision_scores[index_max]
    results["accuracy"] = accuracy_scores[index_max]
    best_param_value = param_values[index_max]
    return best_param_value, results
    
def get_best_parameters_with_results(param_grid_wordembeddings, param_grid_training, score):
    """ Get the best parameter values for tuning wordembeddings as well as for tuning training model,
        along with the metric results

    Parameters
    ----------
    param_grid_wordembeddings: dictionary of parameter:[list of values] for tuning wordembeddings
    param_grid_training: dictionary of parameter:[list of values] for tuning training model
    score: accepts following: 'auprc', 'auc', 'recall', 'precision', 'accuracy',
           depending on which evaluation metric we want to apply

    Returns
    -------
    Returns all best parameter values (evaluated based on metric score 'score') for wordembeddings as well as for training (both as dict)
    and the evaluation results (auprc, auc, recall, precision and accuracy) when applying these best parameter values (as dict)

    """
    best_params_wordembeddings = {}
    best_params_training = {}
    final_results = {}
    for param in param_grid_wordembeddings:
        param_values = param_grid_wordembeddings[param]
        tune = "wordembeddings"
        best_param_value, results = get_best_parameter_value_with_results(param, param_values, best_params_wordembeddings, best_params_training, tune, score, X_test, y_test)
        best_params_wordembeddings[param] = best_param_value
    for param in param_grid_training:
        param_values = param_grid_training[param]
        tune = "training"
        best_param_value, results = get_best_parameter_value_with_results(param, param_values, best_params_wordembeddings, best_params_training, tune, score, X_test, y_test)
        best_params_wordembeddings[param] = best_param_value
        if len(best_params_wordembeddings) == len(param_grid_training):# we are in the last iteration, so we want to show the evaluation results with all (best) parameters tuned
            final_results = results
    return best_params_wordembeddings, best_params_training, final_results




################# loading data
df_all = pd.read_csv('./input/merged_file_all.csv', sep='\t', encoding = 'utf-8')
df_all = df_all[['labels', 'project_details']].copy()
df_all1 = df_all

################# prepare data for fasttext text representation 
## make data compatible for fasttext 
## data.txt shape:
# __label__0 "some text..."  (pre-processed project details)
# __label__1 "some text..."
# __label__0 "some text..."

data_file = "data.txt"
with io.open(data_file,'w',encoding='utf8') as f:
    for i in range(0,len(df_all['labels'])):
        f.write("__label__" + str(df_all['labels'][i]) + " " + df_all['project_details'][i] + "\n")



################## prepare data for fasttext text classification
# Validation Set approach : take 75% of the data as the training set and 25 % as the test set. X is a dataframe with  the input variable
# K fold cross-validation approach as well?
length_to_split = int(len(df_all) * 0.75)

X = df_all['project_details']
y = df_all['labels']

## Splitting the X and y into train and test datasets
X_train, X_test = X[:length_to_split], X[length_to_split:]
y_train, y_test = y[:length_to_split], y[length_to_split:]
#conversion
y_train = y_train.tolist()
X_train = X_train.tolist()
y_test = y_test.tolist()
X_test = X_test.tolist()

## make number of yes == number of no
# take the difference of number of yes and no labels
diff = abs(y_test.count(1)-y_test.count(0))
y_test_new = []
X_test_new = []

c = 0
if y_test.count(1) < y_test.count(0): # basically this case is true, since there are much more 0's than 1's
    for i in range(0,len(y_test)):
        if (y_test[i] == 1) or (y_test[i] == 0 and c >= diff):
            y_test_new.append(y_test[i])
            X_test_new.append(X_test[i])
        elif y_test[i] == 0 and c < diff:
            c = c + 1
elif y_test.count(0) < y_test.count(1): 
    for i in range(0,len(y_test)):
        if (y_test[i] == 0) or (y_test[i] == 1 and c >= diff):
            y_test_new.append(y_test[i])
            X_test_new.append(X_test[i])
        elif y_test[i] == 1 and c < diff:
            c = c + 1

y_test_old = y_test
X_test_old = X_test
y_test = y_test_new
X_test = X_test_new

## make train and test set compatible for fastText
# ...
## train_set.txt shape:
# __label__0 "some text..."  (pre-processed project details)
# __label__1 "some text..."
# __label__0 "some text..."

train_file = "train_set.txt"  
test_file = "test_set.txt"  
with io.open(train_file,'w',encoding='utf8') as f:
    for i in range(0,len(y_train)):
        f.write("__label__" + str(y_train[i]) + " " + X_train[i] + "\n")

with io.open(test_file,'w',encoding='utf8') as f:
    for i in range(0,len(y_test)):
        f.write("__label__" + str(y_test[i]) + " " + X_test[i] + "\n")



################## Apply Fastext and fine tune the parameters for better results
'''
We consider two main steps for better results:
- tuning parameters for text representation (word embeddings)
- tuning parameters for training
The optimal classification threshold will also be applied
'''
####### tuning parameters for fasttext WORD EMBEDDINGS
'''
    input             # training file path (required)
    model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
    lr                # learning rate [0.05]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [5]
    minn              # min length of char ngram [3]
    maxn              # max length of char ngram [6]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [ns]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    verbose           # verbose [2]
'''
## Tutorial for word representations suggests to tune especially following parameters:
# - dim = [100,300]
# - minn and maxn
# - epoch (if dataset is massive, better < 5)
# - lr = [0.01,1.0] 

dim = [100,125,150,175,200,225,250,275,300]
minn = [2,3,4]
maxn = [5,6,7]
epoch = [1,2,3,4,5,6,7]
lr = list(np.arange(0.1, 1.0, 0.05))
param_grid_wordembeddings= {"dim":dim, "minn":minn, "maxn":maxn, "epoch":epoch, "lr":lr}

####### tuning parameters for fasttext TRAINING
'''
    input             # training file path (required)
    lr                # learning rate [0.1]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [1]
    minCountLabel     # minimal number of label occurences [1]
    minn              # min length of char ngram [0]
    maxn              # max length of char ngram [0]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [softmax]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    label             # label prefix ['__label__']
    verbose           # verbose [2]
    pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
'''
## Tutorial for text classification suggests to tune especially following parameters:
# - epoch [5,50]
# - lr = [0.01,1.0] 
# - wordNgrams [1,5]
# And pre-processing the data (already done in another step)

# Dimension of look-up table in training model has to be equal to dimension of embeddings,
# get_best_parameter_value_with_results function handles this issue
epoch = list(np.arange(5, 50, 5))
lr = list(np.arange(0.1, 1.0, 0.05))
wordNgrams = [1,2,3,4,5]
param_grid_training= {"epoch":epoch, "lr":lr, "wordNgrams":wordNgrams}
score = "auprc" 

best_params_wordembeddings, best_params_training, final_results = get_best_parameters_with_results(param_grid_wordembeddings, param_grid_training, score)
print("Best word embeddings parameter values: \n")
for param in best_params_wordembeddings:
    best_value = best_params_wordembeddings[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")
print("Best training parameter values: \n")
for param in best_params_training:
    best_value = best_params_training[param]
    print(param + " : " + str(best_value) + "\n")
print("\n")
print("Final evaluation results: \n")
for metric in final_results:
    result = final_results[metric]
    print(metric + " : " + str(result) + "\n")



'''
path, dirs, files = next(os.walk("evaluations/model_fasttext_fasttext/tune_" + tune))
dirs_count = len(dirs)
evaluation_file = "evaluations/model_fasttext_fasttext/param_set_" + str(dirs_count + 1) + "/" + param + ".txt"
with io.open(data_file,'w+',encoding='utf8') as f:
    f.write("Tune " + tune + " parameter " + param + " on values: \n")
    f.write(str(param_values) + "\n\n")
    f.write("with following parameter settings: \n")
    f.write("with following parameter settings: \n")
    
    for i in range(0,len(df_all['labels'])):
        f.write("__label__" + str(df_all['labels'][i]) + " " + df_all['project_details'][i] + "\n")
'''
