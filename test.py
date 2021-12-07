from preprocess import data_frame_reg_to_cls
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from evaluation import test_task

TASK = 'cls'

D0 = pd.read_csv("ionosphere_data.csv")
D0 =D0.replace({'column_ai': {'g': 1,'b':0}})
X = D0.iloc[:, :-1]
y = D0.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# X_train.to_csv("german_credit_X_train.csv",index=False)
# X_test.to_csv("german_credit_X_test.csv",index=False)
# y_train.to_csv("german_credit_y_train.csv",index=False)
# y_test.to_csv("german_credit_y_test.csv",index=False)


clf = RandomForestClassifier(random_state=0).fit(X_train,y_train)
y_predict = clf.predict(X_test)
print(accuracy_score(y_test,y_predict),precision_score(y_test,y_predict),recall_score(y_test,y_predict),f1_score(y_test,y_predict))