# https://stackoverflow.com/questions/28716241/controlling-the-threshold-in-logistic-regression-in-scikit-learn
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import os
import numpy as np
import json
import re
import ast
from nltk.tokenize import word_tokenize

import json
import pandas as pd

import os

import numpy as np
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


import glob, os
os.chdir('./projects_ID_NO')
json_files = [file for file in glob.glob('*.json')]
path = './projects_ID_NO/'
files = [path + f for f in json_files]
# go back
os.chdir('..')

#1 from 0 to 5000    0
#2 from 5000 to 10000  1
#3 from 10000 to 15000  2
#4 from 15000 to 20000  3
#5 from 20000 to 25000  4
#6 from 25000 to 30000  5
#7 from 30000 to 35000  6
#8 from 35000 to 40000  7
#9 from 40000 to 45000   8
#10 from 45000 to 50000   9
#11 from 50000 to 55000   10
#12 from 55000 to 60000   11
#13 from 60000 to len(files)  (= around 63000)  12

dict_list = []
### work with the labeled projects     
for m in range(60000,len(files)):
        # read file
        with open(files[m], 'r') as f:
            data=f.read()
            # parse file
            j_data = json.loads(data)
            dict_list.append(j_data)
# Convert list of dicts into dataframe and send to csv

# merged_file1.csv 0
# merged_file2.csv 1
# merged_file3.csv 2
# merged_file4.csv  3
# merged_file5.csv  4
# merged_file6.csv  5
# merged_file7.csv  6
# merged_file8.csv  7
# merged_file9.csv  8
# merged_file10.csv  9
# merged_file11.csv  10
# merged_file12.csv 11
# merged_file13.csv  12
df = pd.DataFrame(dict_list)
df.to_csv('./input/merged_file13.csv', sep='\t', encoding = 'utf-8')
# loading data
df = pd.read_csv('./input/merged_file13.csv',sep='\t', encoding = 'utf-8')
df.shape  # (# items (rows), # features (columns))


# Create a new dataframe with two columns
df1 = df[['label', 'project_details']].copy()

# Create a new column 'category_id' with encoded categories 
df1['category_id'] = df1['label'].factorize()[0]
df2 = df1

category_id_df = df1[['label', 'category_id']].drop_duplicates()


########################### CLEANING STEP #############################################


f = open("./stopwords/sw_it.txt","r")
text = f.read()
# convert string representation of list into list
useless_it = ast.literal_eval(text)

f = open("./stopwords/sw_de.txt","r")
text = f.read()
# convert string representation of list into list
useless_de = ast.literal_eval(text)

f = open("./stopwords/sw_fr.txt","r")
text = f.read()
# convert string representation of list into list
useless_fr = ast.literal_eval(text)

f = open("./stopwords/sw_en.txt","r")
text = f.read()
# convert string representation of list into list
useless_en = ast.literal_eval(text)


useless_all = useless_it + useless_de + useless_fr + useless_en
# unaccent the words to make it language independent
useless_all = [unidecode.unidecode(x) for x in useless_all]
# drop duplicates
useless_all = list(dict.fromkeys(useless_all))

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


mistakes = ['\n']
actuals = [' ']
rep = {}

# cleaning mistakes 
for i in range(0,len(mistakes)) :
    rep[mistakes[i]] = actuals[i]

# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in rep.items()) 
#Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
pattern = re.compile("|".join(rep.keys()))


punctuations = {'«','»','!','(',')','-','[',']','{','}',';',':','\'','\"','\\',',','<','>','.','/','?','@','#','$','%','^','&','*','_','~', "''", '``', "'", "’", ' '}
punctuations_str = '''«»!()-[]{};:'"\,<>./?@#$%^&*_~''``'’'''
translator = str.maketrans(punctuations_str, ' '*len(punctuations_str))

failed_extraction = []
all_projects = {}
all_projects['project_details'] = []
all_projects['label'] = []
all_projects['category_id'] = []

for i in range(0,len(df1['project_details'])):
    text = df1['project_details'][i]
    text_raw = str(text)
    label = str(df1['label'][i])
    cat = str(df1['category_id'][i])
    if text == '{}':
        failed_extraction.append(i)
        continue
    # remove unwanted substrings
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    # remove punctuation from text and replace by space
    text = str(text.translate(translator))
    # lower text
    text = text.lower()
    # unaccent the words to make it language independent
    text = unidecode.unidecode(text)
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    # split into tokens
    tokens = text.split()
    filtered_sentence = []
    for w in tokens:
        if (w not in useless_all and not hasNumbers(w) and len(w)>1): 
            filtered_sentence.append(w)
    df2['project_details'][i] = ' '.join(filtered_sentence)
    all_projects['project_details'].append(df2['project_details'][i])
    all_projects['label'].append(label)
    all_projects['category_id'].append(cat)


####### save the cleaned results, cause takes long time


#1 from 0 to 5000   0
#2 from 5000 to 10000  1  
#3 from 10000 to 15000  2
#4 from 15000 to 20000   3
#5 from 20000 to 25000  4
#6 from 25000 to 30000  5
#7 from 30000 to 35000  6
#8 from 35000 to 40000  7
#9 from 40000 to 45000  8
#10 from 45000 to 50000  9
#11 from 50000 to 55000  10
#12 from 55000 to 60000  11
#13 from 60000 to len(files)  12

#1 from 0 to 10000   0
#2 from 10000 to 20000  1  
#3 from 20000 to 30000  2
#4 from 30000 to 40000   3
#5 from 40000 to 50000  4
#6 from 50000 to 60000  5
#7 from 60000 to len(files)  6


directory = './lang_projects'
### 1
# all
with open(directory + '/all_13.json', 'w') as outfile:
    json.dump(all_projects, outfile)


################# collect all cleaned data together

json_files = [filename for filename in os.listdir('./lang_projects')]
all_list = [filename for filename in json_files if filename.startswith("all")]

path = './lang_projects/'
all_list = [path + f for f in all_list]

dict_all_list = []

for file in all_list:
    # read file
    with open(file, 'r') as f:
        data=f.read()
        # parse file
        j_data = json.loads(data)
        dict_all_list.append(j_data)

# convert list of dicts into list of project_details

# de_1.json is a list of lists of strings
# others are a dictionary with a single key 'project_details' representing a list of strings
# why??

all_projects_list = []
all_cat_list = []
all_label_list = []
for d in range(0,len(dict_all_list)):
    projects = dict_all_list[d]['project_details']
    all_projects_list.extend(projects)
    #c = dict_all_list[d]['category_id']
    #all_cat_list.extend(c)
    labels = dict_all_list[d]['label']
    labels1 = []
    for l in labels:
        if l == 'yes1' or l == 'yes2':
            l1 = 'yes'
        else:
            l1 = 'no'
        labels1.append(l1)
    all_label_list.extend(labels1)


#d = {'project_details':all_projects_list,'category_id':all_cat_list, 'labels':all_label_list}
d = {'project_details':all_projects_list, 'labels':all_label_list}
d['category_id'] = list(pd.factorize(d['labels'])[0])

# Convert list of dicts into dataframe and send to csv
df_all = pd.DataFrame(d)
for i in range(0, len(df_all['labels'])):
    if df_all['labels'][i] == 'yes':
        df_all['labels'][i] = 1
    else:
        df_all['labels'][i] = 0
df_all.to_csv('./input/merged_file_all.csv', sep='\t', encoding = 'utf-8')

# loading data
df_all = pd.read_csv('./input/merged_file_all.csv', sep='\t', encoding = 'utf-8')
df_all1 = df_all

########################### TFIDF STEP #############################################
############### 2 - Use TF-IDF to convert the textual data to vectors:  ##################
'''
- For this step, you can use TF-IDF from sickit-learn. You need to try two options:
a- apply TF-IDF on all projects regardless of the language 
b- Divide the projects you have based on the language (group 1 french, group2 german…) 
    and apply TF-IDF on each group seperately

TfidfVectorizer class can be initialized with the following parameters:

min_df: remove the words from the vocabulary which have occurred in less than ‘min_df’ number of files.
   (float in range [0.0, 1.0] or int (default=1))
   When building the vocabulary ignore terms that have a document frequency strictly lower
   than the given threshold. This value is also called cut-off in the literature.
   If float, the parameter represents a proportion of documents, integer absolute counts.
   This parameter is ignored if vocabulary is not None.
max_df: remove the words from the vocabulary which have occurred in more than _‘maxdf’ * total number of files in corpus.
   (float in range [0.0, 1.0] or int (default=1.0))
   When building the vocabulary ignore terms that have a document frequency strictly higher
   than the given threshold (corpus-specific stop words). If float, the parameter represents
   a proportion of documents, integer absolute counts. This parameter is ignored
   if vocabulary is not None
sublinear_tf: set to True to scale the term frequency in logarithmic scale.
stop_words: remove the predefined stop words in 'english'.
use_idf: weight factor must use inverse document frequency.
ngram_range: (1, 2) to indicate that unigrams and bigrams will be considered.
'''

### a- apply TF-IDF on all projects regardless of the language
# magic_nr_all = int(0.2*len(df2.project_details))
'''
tfidf_all = TfidfVectorizer(sublinear_tf=True, min_df=0.2, max_df=0.9) # ngram_range=(1, 2) for unigrams and bigrams
features_all = tfidf_all.fit_transform(df2.project_details).toarray()

tfidf_all1 = TfidfVectorizer(sublinear_tf=True, min_df=0.2, max_df=0.85) # ngram_range=(1, 2) for unigrams and bigrams
features_all1 = tfidf_all1.fit_transform(df2.project_details).toarray()
'''
tfidf_all2 = TfidfVectorizer(sublinear_tf=True, min_df=0.001, max_df=0.85) # min_df=0.3, max_df=0.9
features_all2 = tfidf_all2.fit_transform(df_all['project_details']).toarray()




############################ Classification #############################
'''
Multi-Classification models:
The classification models evaluated are:

- Random Forest
- Linear Support Vector Machine
- Multinomial Naive Bayes
- Logistic Regression.

Spliting the data into train and test sets
The original data was divided into features (X) and target (y),
which were then splitted into train (75%) and test (25%) sets.
Thus, the algorithms would be trained on one set of data and tested out
on a completely different set of data (not seen before by the algorithm).

'''
'''
# all 
x_all = df_all['project_details'] # Collection of documents
y_all = df_all['labels'] # Target or the labels we want to predict 

x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x_all, y_all, 
                                                    test_size=0.25,
                                                    random_state = 0)

models = [
    #RandomForestClassifier(n_estimators=100,criterion="gini" max_depth=5, random_state=0),
    RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

#### 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features_all2, df_all['labels'], scoring='accuracy', cv=CV)  # best tfidf results with all2
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


#### Comparison of model performance
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)  
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc
# --> best results give RandomForestClassifier and LogisticRegression, both with 0.835714 accuracy

'''


###### Model evaluation


#X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features_all2, df_all['labels'], df_all.index, test_size=0.25, random_state=1)

####  Hyperparameter tuning #########
'''

#### test1

##logreg
param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge  # for logreg
mod=LogisticRegression()

##logreg1
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
class_weight = ['balanced']
solver = ['liblinear', 'saga']
param_grid = {"C":C, "penalty":penalty, "class_weight":class_weight}# l1 lasso l2 ridge
mod=LogisticRegression()


##multinomialNB
param_grid = {
  'alpha': np.linspace(0.5, 1.5, 6),
  'fit_prior': [True, False],  
}
mod=MultinomialNB()


##randomforest
n_estimators = [5, 10, 20, 30, 50]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
param_grid = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
mod=RandomForestClassifier()


##linearsvc
penalty = ['l1', 'l2']
C = np.logspace(-3,3,7)
param_grid  = dict(C=C, penalty=penalty)
mod=LinearSVC()


##linearsvc1
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
class_weight = ['balanced']
param_grid = {"C":C, "penalty":penalty, "class_weight":class_weight}# l1 lasso l2 ridge
mod=LinearSVC()

grid = GridSearchCV(mod,param_grid,cv=10)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

OUTPUT:
- logreg: Best Score:  0.9547076467889328, Best Params:  {'C': 100.0, 'penalty': 'l2'}
- logreg1: Best Score: 0.9508, Best Params: {'penalty': 'l1', 'class_weight': 'balanced', 'solver': 'liblinear' 'C': 10.0}
- multinomialnb: Best Score:  0.8346576659349404, Best Params:  {'fit_prior': True, 'alpha': 1.5}
- linearsvc: Best Score:  0.9547733689575566, Best Params:  {'penalty': 'l2', 'C': 1.0}
- linearsvc1: Best Score:  0.9472, Best Params: {'penalty': 'l2', 'class_weight': 'balanced' 'C': 10}
'''

#https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features_all2, df_all['labels'], df_all.index, test_size=0.25, random_state=1)

#conversion
y_test = y_test.tolist()
X_test = X_test.tolist()

#### number of yes == number of no
# y_test.count('0')
# output: 13404
# y_test.count('1')
# output: 1823
# take the difference of number of yes and no labels
diff = abs(y_test.count(1)-y_test.count(0))
y_test_new = []
X_test_new = []

c = 0
for i in range(0,len(y_test)):
    if y_test[i] == 1:
        y_test_new.append(y_test[i])
        X_test_new.append(X_test[i])
    elif y_test[i] == 0 and c < diff:
        c = c + 1
        # don't add to list
    elif y_test[i] == 0 and c >= diff:
        y_test_new.append(y_test[i])
        X_test_new.append(X_test[i])

y_test = y_test_new
X_test = X_test_new


model_mnb = MultinomialNB(fit_prior= True, alpha= 1.5)
model_rf =  RandomForestClassifier(n_estimators=50,criterion="entropy", class_weight = "balanced")
model_lr = LogisticRegression(C= 100.0, penalty= 'l2')
model_lr1 = LogisticRegression(C= 10.0, penalty= 'l1', class_weight= 'balanced', solver='liblinear')
model_lsvc = LinearSVC(C= 1.0, penalty= 'l2')
model_lsvc1 = LinearSVC(C= 10.0, penalty= 'l2', class_weight= 'balanced')


model_mnb.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_lr1.fit(X_train, y_train)
model_lsvc.fit(X_train, y_train)
model_lsvc1.fit(X_train, y_train)


# predict class labels for the test set. The predict function converts probability values > .5 to 1 else 0

y_pred_mnb = model_mnb.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)
y_pred_lr1 = model_lr1.predict(X_test)
y_pred_lsvc = model_lsvc.predict(X_test)
y_pred_lsvc1 = model_lsvc1.predict(X_test)


# generate class probabilites
# Notice that 2 elements will be returned in probs array,
# 1st element is probability for negative class,
# 2nd element gives probability for positive class
probs_mnb = model_mnb.predict_proba(X_test)
probs_rf = model_rf.predict_proba(X_test)
probs_lr = model_lr.predict_proba(X_test)
probs_lr1 = model_lr1.predict_proba(X_test)

y_scores_mnb = probs_mnb[:,1]
y_scores_rf = probs_rf[:,1]
y_scores_lr = probs_lr[:,1]
y_scores_lr1 = probs_lr1[:,1]
y_scores_lsvc = model_lsvc.decision_function(X_test)
y_scores_lsvc1 = model_lsvc1.decision_function(X_test)

# generate evaluation metrics
print("Accuracy MNB: ", metrics.accuracy_score(y_test, y_pred_mnb))
print("Accuracy RF: ", metrics.accuracy_score(y_test, y_pred_rf))
print("Accuracy LR: ", metrics.accuracy_score(y_test, y_pred_lr))
print("Accuracy LR1: ", metrics.accuracy_score(y_test, y_pred_lr1))
print("Accuracy LSVC: ", metrics.accuracy_score(y_test, y_pred_lsvc))
print("Accuracy LSVC1: ", metrics.accuracy_score(y_test, y_pred_lsvc1))

# Find optimal cutoff point
# The optimal cut-off would be where the true positive rate (tpr) is high
# and the false positive rate (fpr) is low,
# and tpr (- fpr) is zero or near to zero
# Plot a ROC of tpr vs 1-fpr

# extract fpr and tpr
fpr_mnb, tpr_mnb, thresholds_mnb = metrics.roc_curve(y_test, y_scores_mnb) 
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_test, y_scores_rf)
fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test, y_scores_lr)
fpr_lr1, tpr_lr1, thresholds_lr1 = metrics.roc_curve(y_test, y_scores_lr1)
fpr_lsvc, tpr_lsvc, thresholds_lsvc = metrics.roc_curve(y_test, y_scores_lsvc)
fpr_lsvc1, tpr_lsvc1, thresholds_lsvc1 = metrics.roc_curve(y_test, y_scores_lsvc1)

auc_mnb = metrics.auc(fpr_mnb, tpr_mnb)
auc_rf = metrics.auc(fpr_rf, tpr_rf)
auc_lr = metrics.auc(fpr_lr, tpr_lr)
auc_lr1 = metrics.auc(fpr_lr1, tpr_lr1)
auc_lsvc = metrics.auc(fpr_lsvc, tpr_lsvc)
auc_lsvc1 = metrics.auc(fpr_lsvc1, tpr_lsvc1)


plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr_mnb, tpr_mnb, linestyle='-', label= 'MNB (auc = %0.3f)' % auc_mnb)
plt.plot(fpr_rf, tpr_rf, linestyle='-', label= 'RF (auc = %0.3f)' % auc_rf)
plt.plot(fpr_lr, tpr_lr, linestyle='-', label= 'LR (auc = %0.3f)' % auc_lr)
plt.plot(fpr_lr1, tpr_lr1, linestyle='-', label= 'LR1 (auc = %0.3f)' % auc_lr1)
plt.plot(fpr_lsvc, tpr_lsvc, linestyle='-', label= 'LSVC (auc = %0.3f)' % auc_lsvc)
plt.plot(fpr_lsvc1, tpr_lsvc1, linestyle='-', label= 'LSVC1 (auc = %0.3f)' % auc_lsvc1)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend(loc='lower right')
plt.savefig('auc_graph8.png')

def roc_model(tpr,fpr, thresholds):
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
    'tpr' : pd.Series(tpr, index = i), 
    '1-fpr': pd.Series(1-fpr, index=i), 
    'tf': pd.Series(tpr - (1-fpr), index=i), 
    'thresholds' : pd.Series(thresholds, index=i)})
    roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return roc
    

roc_mnb = roc_model(tpr_mnb,fpr_mnb, thresholds_mnb)
roc_rf = roc_model(tpr_rf, fpr_rf, thresholds_rf)
roc_lr = roc_model(tpr_lr, fpr_lr, thresholds_lr)
roc_lr1 = roc_model(tpr_lr1, fpr_lr1, thresholds_lr1)
roc_lsvc = roc_model(tpr_lsvc, fpr_lsvc, thresholds_lsvc)
roc_lsvc1 = roc_model(tpr_lsvc1, fpr_lsvc1, thresholds_lsvc1)

# Plot tpr vs 1-tpr
fig, ax = plt.subplots()
#plt.plot(roc_mnb['tpr'], label='tpr mnb')
#plt.plot(roc_mnb['1-fpr'], color = 'red', label= '1-fpr mnb')
#plt.plot(roc_rf['tpr'], label='tpr rf')
#plt.plot(roc_rf['1-fpr'], color = 'red', label= '1-fpr rf')
#plt.plot(roc_lr['tpr'], label='tpr lr')
#plt.plot(roc_lr['1-fpr'], color = 'red', label= '1-fpr lr')
plt.plot(roc_lr1['tpr'], label='tpr lr1')
plt.plot(roc_lr1['1-fpr'], color = 'red', label= '1-fpr lr1')
plt.plot(roc_lsvc['tpr'], label='tpr lsvc')
plt.plot(roc_lsvc['1-fpr'], color = 'red', label= '1-fpr lsvc')
#plt.plot(roc_lsvc1['tpr'], label='tpr lsvc1')
#plt.plot(roc_lsvc1['1-fpr'], color = 'red', label= '1-fpr lsvc1')
plt.legend(loc='best')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receive operating characteristic')
plt.savefig('auc_graph9.png')

# From the chart, the point where 'tpr' crosses '1-fpr' is the optimal cutoff point.
# To simplify finding the optimal probability threshold and enabling reusability,
# We make a function to find the optimal probability cutoff point

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

# Find optimal probability threshold
# Note: probs[:,1] will have the probability of being positive label
opt_threshold_lr1 =find_optimal_cutoff(y_test, y_scores_lr1)
opt_threshold_lsvc =find_optimal_cutoff(y_test, y_scores_lsvc)

# Applying a specific threshold to the prediction probability
threshold = 0.1
y_pred_lr1_1 = np.where(y_scores_lr1 >= threshold, 1,0)
print(metrics.confusion_matrix(y_test,y_pred_lr1_1))
print(metrics.accuracy_score(y_test,y_pred_lr1_1))
threshold = -0.619412
y_pred_lsvc_1 = np.where(y_scores_lsvc>= threshold, 1,0)
print(metrics.confusion_matrix(y_test,y_pred_lsvc_1))
print(metrics.accuracy_score(y_test,y_pred_lsvc_1))







"""
mnb_fpr, mnb_tpr, mnb_threshold = roc_curve(y_test, y_pred_mnb)
auc_mnb = auc(mnb_fpr, mnb_tpr)

rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, y_pred_rf)
auc_rf = auc(rf_fpr, rf_tpr)

lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, y_pred_lr)
auc_lr = auc(lr_fpr, lr_tpr)

lr1_fpr, lr1_tpr, lr1_threshold = roc_curve(y_test, y_pred_lr1)
auc_lr1 = auc(lr1_fpr, lr1_tpr)

lsvc_fpr, lsvc_tpr, threshold = roc_curve(y_test, y_pred_lsvc)
auc_lsvc = auc(lsvc_fpr, lsvc_tpr)

lsvc1_fpr, lsvc1_tpr, lsvc1_threshold = roc_curve(y_test, y_pred_lsvc1)
auc_lsvc1 = auc(lsvc1_fpr, lsvc1_tpr)

plt.figure(figsize=(5,5), dpi=100)
plt.plot(mnb_fpr, mnb_tpr, linestyle='-', label= 'MNB (auc = %0.3f)' % auc_mnb)
plt.plot(rf_fpr, rf_tpr, linestyle='-', label= 'RF (auc = %0.3f)' % auc_rf)
plt.plot(lr_fpr, lr_tpr, linestyle='-', label= 'LR (auc = %0.3f)' % auc_lr)
plt.plot(lr1_fpr, lr1_tpr, linestyle='-', label= 'LR1 (auc = %0.3f)' % auc_lr1)
plt.plot(lsvc_fpr, lsvc_tpr, linestyle='-', label= 'LSVC (auc = %0.3f)' % auc_lsvc)
plt.plot(lsvc1_fpr, lsvc1_tpr, linestyle='-', label= 'LSVC1 (auc = %0.3f)' % auc_lsvc1)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend(loc='lower right')
plt.savefig('auc_graph2.png')



## Threshold representation (which is best threshold?) with precision recall
# False positive rate != Precision, True positive rate != Recall


####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])



### adjust the threshold
# predict_proba predicts the probability of each class for a datapoint
y_pred1_mnb = model_mnb.predict_proba(X_test)
y_pred1_rf = model_rf.predict_proba(X_test)
y_pred1_lr = model_lr.predict_proba(X_test)
y_pred1_lr1 = model_lr1.predict_proba(X_test)
y_pred1_lsvc = model_lsvc.decision_function(X_test)  # svc has no predict_proba
y_pred1_lsvc1 = model_lsvc1.decision_function(X_test) # svc has no predict_proba

y_scores_mnb = y_pred1_mnb[:,1]
y_scores_rf = y_pred1_rf[:,1]
y_scores_lr = y_pred1_lr[:,1]
y_scores_lr1 = y_pred1_lr1[:,1]
y_scores_lsvc = y_pred1_lsvc[:,1]
y_scores_lsvc1 = y_pred1_lsvc1[:,1]

mnb_precsion_2, mnb_recall_2, mnb_thresholds_2 = precision_recall_curve(y_test, y_scores_mnb)
rf_precsion_2, rf_recall_2, rf_thresholds_2 = precision_recall_curve(y_test, y_scores_rf)
lr_precsion_2, lr_recall_2, lr_thresholds_2 = precision_recall_curve(y_test, y_scores_lr)
lr1_precsion_2, lr1_recall_2, lr1_thresholds_2 = precision_recall_curve(y_test, y_scores_lr1)
lsvc_precsion_2, lsvc_recall_2, lsvc_thresholds_2 = precision_recall_curve(y_test, y_scores_lsvc)
lsvc1_precsion_2, lsvc1_recall_2, lsvc1_thresholds_2 = precision_recall_curve(y_test, y_scores_lsvc1)


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, y_scores, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)

    '''
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    '''
    print(metrics.classification_report(y_test, y_pred_adj,target_names= ['0','1']))

    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)





'''

plt.figure(figsize=(5,5), dpi=100)
plt.title("False Postive rate - True Positive Rate vs Threshold Chart")
plt.plot(mnb_thresholds_1, mnb_fpr_1, "b--", label="MNB FPR")
plt.plot(mnb_thresholds_1, mnb_tpr_1, "r--", label="MNP TPR")
plt.plot(rf_thresholds_1, rf_fpr_1, "b--", label="RF FPR")
plt.plot(rf_thresholds_1, rf_tpr_1, "r--", label="RF TPR")
plt.plot(lr_thresholds_1, lr_fpr_1, "b--", label="LR FPR")
plt.plot(lr_thresholds_1, lr_tpr_1, "r--", label="LR TPR")
plt.plot(lr1_thresholds_1, lr1_fpr_1, "b--", label="LR1 FPR")
plt.plot(lr1_thresholds_1, lr1_tpr_1, "r--", label="LR1 TPR")
plt.ylabel("FPR, TPR")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])
plt.savefig('auc_graph7.png')
'''


# Precision, Recall, F1-score
# Classification report
y_test = [str(i) for i in y_test]  # y_test already is a list
y_pred_mnb = [str(i) for i in y_pred_mnb.tolist()]
y_pred_rf = [str(i) for i in y_pred_rf.tolist()]
y_pred_lr = [str(i) for i in y_pred_lr.tolist()]
y_pred_lr1 = [str(i) for i in y_pred_lr1.tolist()]
y_pred_lsvc = [str(i) for i in y_pred_lsvc.tolist()]
y_pred_lsvc1 = [str(i) for i in y_pred_lsvc1.tolist()]

print('\t\t\t\tCLASSIFICATIION METRICS\n')

print(metrics.classification_report(y_test, y_pred_mnb,target_names= ['0','1']))

print(metrics.classification_report(y_test, y_pred_rf,target_names= ['0','1']))

print(metrics.classification_report(y_test, y_pred_lr,target_names= ['0','1']))

print(metrics.classification_report(y_test, y_pred_lr1,target_names= ['0','1']))

print(metrics.classification_report(y_test, y_pred_lsvc,target_names= ['0','1']))

print(metrics.classification_report(y_test, y_pred_lsvc1,target_names= ['0','1']))







'''
#### number of yes == number of no

# Precision, Recall, F1-score
# Classification report

# y_test.count('0')
# output: 13404
# y_test.count('1')
# output: 1823

# take the difference of number of yes and no labels
diff = abs(y_test.count('1')-y_test.count('0'))
y_test_new = []
y_pred_mnb_new = [] 
y_pred_rf_new = []
y_pred_lr_new = []
y_pred_lr1_new = []
y_pred_lsvc_new = []
y_pred_lsvc1_new = []

c = 0
for i in range(0,len(y_test)):
    if y_test[i] == '1':
        y_test_new.append(y_test[i])
        y_pred_mnb_new.append(y_pred_mnb[i])
        y_pred_rf_new.append(y_pred_rf[i])
        y_pred_lr_new.append(y_pred_lr[i])
        y_pred_lr1_new.append(y_pred_lr1[i])
        y_pred_lsvc_new.append(y_pred_lsvc[i])
        y_pred_lsvc1_new.append(y_pred_lsvc1[i])
    elif y_test[i] == '0' and c < diff:
        c = c + 1
        # don't add to list
    elif y_test[i] == '0' and c >= diff:
        y_test_new.append(y_test[i])
        y_pred_mnb_new.append(y_pred_mnb[i])
        y_pred_rf_new.append(y_pred_rf[i])
        y_pred_lr_new.append(y_pred_lr[i])
        y_pred_lr1_new.append(y_pred_lr1[i])
        y_pred_lsvc_new.append(y_pred_lsvc[i])
        y_pred_lsvc1_new.append(y_pred_lsvc1[i])

y_test = y_test_new
y_pred_mnb = y_pred_mnb_new
y_pred_rf = y_pred_rf_new
y_pred_lr = y_pred_lr_new
y_pred_lr1 = y_pred_lr1_new
y_pred_lsvc = y_pred_lsvc_new
y_pred_lsvc1 = y_pred_lsvc1_new

print('\t\t\t\tCLASSIFICATIION METRICS\n')

print(metrics.classification_report(y_test_new, y_pred_mnb_new,target_names= ['0','1']))

print(metrics.classification_report(y_test_new, y_pred_rf_new,target_names= ['0','1']))

print(metrics.classification_report(y_test_new, y_pred_lr_new,target_names= ['0','1']))

print(metrics.classification_report(y_test_new, y_pred_lr1_new,target_names= ['0','1']))

print(metrics.classification_report(y_test_new, y_pred_lsvc_new,target_names= ['0','1']))

print(metrics.classification_report(y_test_new, y_pred_lsvc1_new,target_names= ['0','1']))

# AUC and ROC with predict()
y_test = list(map(int,y_test))
y_pred_mnb = list(map(int,y_pred_mnb))
y_pred_rf = list(map(int, y_pred_rf))
y_pred_lr = list(map(int, y_pred_lr))
y_pred_lr1 = list(map(int, y_pred_lr1))
y_pred_lsvc = list(map(int, y_pred_lsvc))
y_pred_lsvc1 = list(map(int, y_pred_lsvc1))


mnb_fpr, mnb_tpr, mnb_threshold = roc_curve(y_test, y_pred_mnb)
auc_mnb = auc(mnb_fpr, mnb_tpr)

rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, y_pred_rf)
auc_rf = auc(rf_fpr, rf_tpr)

lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, y_pred_lr)
auc_lr = auc(lr_fpr, lr_tpr)

lr1_fpr, lr1_tpr, lr1_threshold = roc_curve(y_test, y_pred_lr1)
auc_lr1 = auc(lr1_fpr, lr1_tpr)

lsvc_fpr, lsvc_tpr, threshold = roc_curve(y_test, y_pred_lsvc)
auc_lsvc = auc(lsvc_fpr, lsvc_tpr)

lsvc1_fpr, lsvc1_tpr, lsvc1_threshold = roc_curve(y_test, y_pred_lsvc1)
auc_lsvc1 = auc(lsvc1_fpr, lsvc1_tpr)

plt.figure(figsize=(5,5), dpi=100)
plt.plot(mnb_fpr, mnb_tpr, linestyle='-', label= 'MNB (auc = %0.3f)' % auc_mnb)
plt.plot(rf_fpr, rf_tpr, linestyle='-', label= 'RF (auc = %0.3f)' % auc_rf)
plt.plot(lr_fpr, lr_tpr, linestyle='-', label= 'LR (auc = %0.3f)' % auc_lr)
plt.plot(lr1_fpr, lr1_tpr, linestyle='-', label= 'LR1 (auc = %0.3f)' % auc_lr1)
plt.plot(lsvc_fpr, lsvc_tpr, linestyle='-', label= 'LSVC (auc = %0.3f)' % auc_lsvc)
plt.plot(lsvc1_fpr, lsvc1_tpr, linestyle='-', label= 'LSVC1 (auc = %0.3f)' % auc_lsvc1)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend(loc='lower right')
plt.savefig('auc_graph3.png')

# AUC and ROC with predict_proba() with #yes == #no
# predict_proba predicts the probability of each class for a datapoint
y_pred1_mnb = model_mnb.predict_proba(X_test)
y_pred1_rf = model_rf.predict_proba(X_test)
y_pred1_lr = model_lr.predict_proba(X_test)
y_pred1_lr1 = model_lr1.predict_proba(X_test)
# y_pred1_lsvc = model_lsvc.predict_proba(X_test) # 'LinearSVC' object has no attribute 'predict_proba'
# y_pred1_lsvc1 = model_lsvc1.predict_proba(X_test) # 'LinearSVC' object has no attribute 'predict_proba'

# take the difference of number of yes and no labels
diff = abs(y_test.count('1')-y_test.count('0'))
y_test_new = []
y_pred_mnb_new = [] 
y_pred_rf_new = []
y_pred_lr_new = []
y_pred_lr1_new = []
#y_pred_lsvc_new = []
#y_pred_lsvc1_new = []

c = 0
for i in range(0,len(y_test)):
    if y_test[i] == '1':
        y_test_new.append(y_test[i])
        y_pred_mnb_new.append(y_pred1_mnb[i])
        y_pred_rf_new.append(y_pred1_rf[i])
        y_pred_lr_new.append(y_pred1_lr[i])
        y_pred_lr1_new.append(y_pred1_lr1[i])
        #y_pred_lsvc_new.append(y_pred1_lsvc[i])
        #y_pred_lsvc1_new.append(y_pred1_lsvc1[i])
    elif y_test[i] == '0' and c < diff:
        c = c + 1
        # don't add to list
    elif y_test[i] == '0' and c >= diff:
        y_test_new.append(y_test[i])
        y_pred_mnb_new.append(y_pred1_mnb[i])
        y_pred_rf_new.append(y_pred1_rf[i])
        y_pred_lr_new.append(y_pred1_lr[i])
        y_pred_lr1_new.append(y_pred1_lr1[i])
        #y_pred_lsvc_new.append(y_pred1_lsvc[i])
        #y_pred_lsvc1_new.append(y_pred1_lsvc1[i])

y_test = y_test_new
y_pred1_mnb = y_pred_mnb_new
y_pred1_rf = y_pred_rf_new
y_pred1_lr = y_pred_lr_new
y_pred1_lr1 = y_pred_lr1_new
#y_pred1_lsvc = y_pred_lsvc_new
#y_pred1_lsvc1 = y_pred_lsvc1_new


#from sklearn.metrics import roc_auc_score
mnb_roc_auc = roc_auc_score(y_test, y_pred_mnb)
rf_roc_auc = roc_auc_score(y_test, y_pred_rf)
lr_roc_auc = roc_auc_score(y_test, y_pred_lr)
lr1_roc_auc = roc_auc_score(y_test, y_pred_lr1)
#lsvc_roc_auc = roc_auc_score(y_test, y_pred_lsvc)
#lsvc1_roc_auc = roc_auc_score(y_test, y_pred_lsvc1)

mnb_fpr, mnb_tpr, mnb_thresholds = roc_curve(y_test, y_pred1_mnb[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, y_pred1_rf[:,1])
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, y_pred1_lr[:,1])
lr1_fpr, lr1_tpr, lr1_thresholds = roc_curve(y_test, y_pred1_lr1[:,1])
lsvc_fpr, lsvc_tpr, lsvc_thresholds = roc_curve(y_test, y_pred1_lsvc[:,1])
lsvc1_fpr, lsvc1_tpr, lsvc1_thresholds = roc_curve(y_test, y_pred1_lsvc1[:,1])

plt.figure(figsize=(5,5), dpi=100)
plt.plot(mnb_fpr, mnb_tpr, linestyle='-', label= 'MNB (auc = %0.3f)' % mnb_roc_auc)
plt.plot(rf_fpr, rf_tpr, linestyle='-', label= 'RF (auc = %0.3f)' % rf_roc_auc)
plt.plot(lr_fpr, lr_tpr, linestyle='-', label= 'LR (auc = %0.3f)' % lr_roc_auc)
plt.plot(lr1_fpr, lr1_tpr, linestyle='-', label= 'LR1 (auc = %0.3f)' % lr1_roc_auc)
plt.plot(lsvc_fpr, lsvc_tpr, linestyle='-', label= 'LSVC (auc = %0.3f)' % lsvc_roc_auc)
plt.plot(lsvc1_fpr, lsvc1_tpr, linestyle='-', label= 'LSVC1 (auc = %0.3f)' % lsvc1_roc_auc)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend(loc='lower right')
plt.savefig('auc_graph4.png')







The classes with more support (number of occurrences) tend
to have a better f1-cscore. This is because the algorithm was trained with more data.
The classe(s) that can be classified with more precision is (are) 'no' .

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.L_decision.values, 
            yticklabels=category_id_df.L_decision.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=8)
plt.show()

# Most correlated terms with each category

model.fit(features_all2, labels_all)

N = 4
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("\n==> '{}':".format(Product))
  print("  * Top unigrams: %s" %(', '.join(unigrams)))
  print("  * Top bigrams: %s" %(', '.join(bigrams)))
'''
"""