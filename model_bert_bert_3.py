from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Load data 
df = pd.read_csv('./input/merged_file_all.csv', sep='\t', encoding = 'utf-8')
df = df[['labels', 'project_details']].copy()

################## prepare data for text classification
# Validation Set approach : take 90% of the data as the training set and 10 % as the test set. X is a dataframe with  the input variable
# K fold cross-validation approach as well?
length_to_split = int(len(df) * 0.90)

X = df['project_details']
y = df['labels']

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



##################  Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.

train_data = [[a,b] for a,b in zip(X_train, y_train)]
train_df = pd.DataFrame(train_data)

eval_data = [[a,b] for a,b in zip(X_test, y_test)]
eval_df = pd.DataFrame(eval_data)


################## Create a ClassificationModel
model = ClassificationModel('distilbert', 'distilbert-base-multilingual-cased', use_cuda=False) # You can set class weights by using the optional weight argument


################## Train the model
model.train_model(train_df)

################## Evaluate the model

result_accuracy, model_outputs_accuracy, wrong_predictions_accuracy = model.eval_model(eval_df, accuracy=sklearn.metrics.accuracy_score)
result_precision, model_outputs_precision, wrong_predictions_precision = model.eval_model(eval_df, precision=sklearn.metrics.precision_score)
result_recall, model_outputs_recall, wrong_predictions_recall = model.eval_model(eval_df, recall=sklearn.metrics.recall_score)
result_auc, model_outputs_auc, wrong_predictions_auc = model.eval_model(eval_df, auc=sklearn.metrics.roc_auc_score)
result_auprc, model_outputs_auprc, wrong_predictions_auprc = model.eval_model(eval_df, auprc=sklearn.metrics.average_precision_score)
