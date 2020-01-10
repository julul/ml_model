# https://stackoverflow.com/questions/44705018/how-to-read-and-combine-multiple-json-files-using-python3
# https://pythonbasics.org/read-json-file/
# https://www.kaggle.com/selener/multi-class-text-classification-tfidf
# https://stackoverflow.com/questions/39142778/python-how-to-determine-the-language
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://www.programiz.com/python-programming/examples/remove-punctuation

import json
import pandas as pd

import os

import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
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
from langdetect import detect

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


dict_list = []

files = ['./projects_ID_NO/ID_54685_NO_947313.json','./projects_ID_NO/ID_116694_NO_935643.json','./projects_ID_NO/ID_122337_NO_1009309.json','./projects_ID_NO/ID_97050_NO_964393.json']

### add decision labels (results from softcom)
m = 1
while(m <= len(files)):
    for file in files:
        # read file
        with open(file) as json_file:
            data = json.load(json_file)
            if (m % 2) == 0:
                data['decision'] = 'yes'
            else:
                data['decision'] = 'no'
        with open(file, 'w') as outfile:
            json.dump(data, outfile)    
        m += 1


### convert json into csv in order to work with the dataset
# read the files with their labels
for file in files:
    # read file
    with open(file, 'r', encoding='utf-8') as f:
        data=f.read()
        # parse file
        j_data = json.loads(data)
        dict_list.append(j_data)

    
# Convert list of dicts into dataframe and send to csv
df = pd.DataFrame(dict_list)
df.to_csv('./input/merged_file.csv', index=False)

print(os.listdir("./input"))

# loading data
df = pd.read_csv('./input/merged_file.csv')
df.shape  # (# items (rows), # features (columns))

# df.head(2).T # Columns are shown in rows for easy reading

'''
The dataset contains features that are not necessary to solve our multi-classification problem.
For this text classification problem, we are going to build another dataframe that contains ‘decision’ and ‘project_details’ (renamed as 'Consumer_complaint').
'''
# Create a new dataframe with two columns
df1 = df[['decision', 'project_details']].copy()

df2 = df1

'''
There are 2 different classes or categories (target): yes or no
Now we need to represent each class as a number, so as our predictive model can better understand the different categories.
'''

# Create a new column 'category_id' with encoded categories 
df1['category_id'] = df1['decision'].factorize()[0]
category_id_df = df1[['decision', 'category_id']].drop_duplicates()

# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'decision']].values)

'''
############ 1 - TEXT PROCESSING #############
- detect language of each text
- remove punctuations
- Remove the stop words (depending on the language detected)
- Move all words to lowercase and remove symbols ( Be careful here for some languages a symbol is a real character 
(e.g. é in english is a symbol that should be replace with e while in french it is a real character!)
'''

'''  
lang0 = detect(df1['project_details'][0]) # de
lang1 = detect(df1['project_details'][1]) # it
lang2 = detect(df1['project_details'][2]) # en
lang3 = detect(df1['project_details'][3]) # fr
'''

### Remove the punctuations
### Remove the stop words depending on the language detected
### Move all words to lowercase

# TODO:
### Remove symbols (Be careful here for some languages a symbol is a real character 
###     (e.g. é in english is a symbol that should be replace with e while in french it is a real character!)

stop_words_en = set(stopwords.words('english')) 
stop_words_it = set(stopwords.words('italian'))
stop_words_de = set(stopwords.words('german'))
stop_words_fr = set(stopwords.words('french'))


for i in range(0,len(df1['project_details'])):
    text = df1['project_details'][i]
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    # remove punctuation from the string
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    text = no_punct
    word_tokens = word_tokenize(text)
    clean_sentence_list = []
    filtered_sentence = []
    if detect(text) == "de":
        for w in word_tokens: 
            if w not in stop_words_de: 
                filtered_sentence.append(w) 
    elif detect(text) == "it":
        for w in word_tokens: 
            if w not in stop_words_it: 
                filtered_sentence.append(w) 
    elif detect(text) == "en":
        for w in word_tokens: 
            if w not in stop_words_en: 
                filtered_sentence.append(w) 
    elif detect(text) == "fr":
        for w in word_tokens: 
            if w not in stop_words_fr: 
                filtered_sentence.append(w)
    lower_sentence = [x.lower() for x in filtered_sentence] 
    # clean_sentence_list.append(lower_sentence)
    df2['project_details'][i] = ' '.join(lower_sentence)

'''
############### 2 - Use TF-IDF to convert the textual data to vectors:  ##################
- For this step, you can use TF-IDF from sickit-learn. You need to try two options:
a- apply TF-IDF on all projects regardless of the language 
b- Divide the projects you have based on the language (group 1 french, group2 german…) 
    and apply TF-IDF on each group seperately
'''
'''
TfidfVectorizer class can be initialized with the following parameters:

min_df: remove the words from the vocabulary which have occurred in less than ‘min_df’ number of files.
max_df: remove the words from the vocabulary which have occurred in more than _‘maxdf’ * total number of files in corpus.
sublinear_tf: set to True to scale the term frequency in logarithmic scale.
stop_words: remove the predefined stop words in 'english'.
use_idf: weight factor must use inverse document frequency.
ngram_range: (1, 2) to indicate that unigrams and bigrams will be considered.
'''

### a- apply TF-IDF on all projects regardless of the language

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))

# We transform each project detail into a vector
## Doesn't take a lists of words, but lists of text
features = tfidf.fit_transform(df2.project_details).toarray()

labels = df2['decision']

print("Each of the %d project details is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))
# Output: Each of the 4 project details is represented by 36 features (TF-IDF score of unigrams and bigrams)
## TODO: What do these multiple features mean? check .fit_transform
## TODO: add other features from json object


### b- Divide the projects you have based on the language (group 1 french, group2 german…) 
###    and apply TF-IDF on each group seperately
# TODO: regarding to stop_words? --> already done


# Finding the three most correlated terms with each of the product categories
N = 3
for decision, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(decision))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))


X = df2['project_details'] # Collection of documents
y = df2['decision'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)



'''
########### 3 - Use Text classification methods on the result you obtain from both options (a and b):
- Many of these classification methods are already implemented in the sklearn library and it is simply one line to call:  #################
- you can try randomforest, svm and Naive Bayes classifier.
Please use this tutorial for reference about this step https://towardsdatascience.com/text-classification-in-python-dd95d264c802
'''



## Models
# TODO: check the different models
#       -> Please use this tutorial for reference about this step https://towardsdatascience.com/text-classification-in-python-dd95d264c802

'''
Multi-Classification models
The classification models evaluated are:

Random Forest
Linear Support Vector Machine
Multinomial Naive Bayes
Logistic Regression.

'''

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]


'''
TODO:
Don't understand well what random_state= 0 means.

random_stateint, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator;
If RandomState instance, random_state is the random number generator;
If None, the random number generator is the RandomState instance used by np.random

'''



# 5 Cross-validation
### TODO: check again well what cross-validation does
CV = 2 # 3, 5, 10 (but cannot be greater than number of samples=4)
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



'''
####### 4- Measure the accuracy, precision, recall and F1 of these methods and report all results in latex (you can use Overleaf) ############
You can also refer to this tutorial: https://stackabuse.com/text-classification-with-python-and-scikit-learn/ for all steps including the preprocessing, tf-idf and the random forest.
'''

### Comparison of model performance
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc

#  -> Best mean accuracy was obtained with LinearSVC (0.5) and LogisticRegression (0.5) 



### Model Evaluation
# -> We choose LinearSVC as model

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, labels, df2.index, test_size=0.25, random_state=1)

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df2.index, test_size=0.25, 
                                                               random_state=1)

# Spliting the data into train and test sets

X = df2['project_details'] # Collection of documents
y = df2['decision'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

    # X features, Y targets. TODO: check what random_state does, and in general check train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)








