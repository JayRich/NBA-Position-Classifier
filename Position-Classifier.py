# Jason Richardson 

!wget -O 2018-2019_NBA_Stats.csv https://uta.instructure.com/courses/61673/files/10505095/preview?verifier=1qoqyIxNj5MjnRrfNGQsUpWHAQs0TDZiyDLfTKps
import io
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

#Read from the csv file and return a Pandas DataFrame.
stats_file = '/content/2018-2019_NBA_Stats.csv'
nba = pd.read_csv(io.FileIO(stats_file))

class_column = 'Position'
 # remove features 'games played' and 'minutes played' 
feature_columns = ['Points Per Game', 'Total Rebounds Per Game',\
                   'Assists Per Game', 'Three Points Made Per Game',\
                   'Steals Per Game', 'Blocks Per Game', 'Turnovers Per Game']


nba_feature = nba[feature_columns]
nba_class = nba[class_column]

# split data into test and train
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, random_state=0)

model = MultinomialNB(0.9)                        # use multinomial naive bayes as model and set hyper-parameter of smoothing 
  
model.fit(train_feature,train_class)            # train the multinomial model on training set

scores = cross_val_score(model, nba_feature, nba_class, cv=5)         # calculate cross validation 
print("Cross-validation scores: {}".format(scores))                   # print scores
print("Average cross-validation score: {:.2f}".format(scores.mean()))   # print average of scores
print()
print("Test set accuracy: {:.2f}".format(model.score(test_feature, test_class)))      # print test accuracy
print()
prediction = model.predict(test_feature)                      # run model on test set
print("Test set predictions:\n{}".format(prediction))
print()
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))   # print confusion matrix
print()
print("Classification report:")
print(classification_report(test_class, prediction))            # calculate and print precision, recall, f-score for classification report

# final model for complete labeled data set

final_model = MultinomialNB(0.9)          # make final multinomial model 

final_model.fit(nba_feature,nba_class)            # train the final multinomial model on complete labeled data for better accuracy
