import json
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

## Read training and test json file 
train_set = pd.read_json('train.json', orient='columns')
test_set = pd.read_json('test.json', orient='columns')

print("Number of records in Training Data: " + str(train_set.shape[0]))
print("Building Decision Tree..")

## Load ingredient and cuisine columns as lists into new columns
train_set['ingreds']= [' , '.join(val) for val in train_set['ingredients']]
train_set['cuisine_fmt'] = [val.strip() for val in train_set['cuisine']]

## Load ingredient columns as list into new column
test_set['ingreds']= [' , '.join(val) for val in test_set['ingredients']]

## Use TFIDF vectorizer to convert the elements into vectors
vectorizer = CountVectorizer()
train_tfidf_ingred = vectorizer.fit_transform(train_set['ingreds'])

## Target class for training set
train_class = train_set['cuisine_fmt']

## Split into train and test sets
X_train, X_test,Y_train,Y_test = train_test_split(train_tfidf_ingred,train_class,test_size=0.10, random_state=42)

## Train data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)

## Predict class for test set (from training data)
Y_test_pred = clf.predict(X_test)
print("Accuracy score of Decision Tree with Count Vectorizer from validation data: " + str(accuracy_score(Y_test,Y_test_pred)))

## Test data to vector
test_tfidf_ingred = vectorizer.transform(test_set['ingreds'])

## Predict the classes of test set of Kaggle
predictions = clf.predict(test_tfidf_ingred)

## Output File
predict_output = pd.DataFrame({'id': test_set['id'], 'cuisine': predictions}, columns=['id', 'cuisine'])

predict_output.to_csv("Whats_Cooking_CountVectorizer.csv",index=False)
