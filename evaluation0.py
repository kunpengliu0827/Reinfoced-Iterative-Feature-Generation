from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def downstream_task(Dg, task='cls', metric='acc'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    if task == 'cls':
        # clf = RidgeClassifier(random_state=0).fit(X_train,y_train)
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if metric == 'acc':
            return accuracy_score(y_test,y_predict)
    if task=='reg':
        reg = Ridge(alpha=1.0).fit(X_train,y_train)
        # reg = DecisionTreeRegressor(random_state=0).fit(X_train,y_train)
        y_predict = reg.predict(X_test)
        if metric == 'mae':
            return mean_absolute_error(y_test,y_predict)

def test_task(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    if task == 'cls':
        # clf = RidgeClassifier(random_state=0).fit(X_train,y_train)
        clf = RandomForestClassifier(random_state=0).fit(X_train,y_train)
        # clf = XGBClassifier(random_state=0).fit(X_train,y_train)
        y_predict = clf.predict(X_test)
        return accuracy_score(y_test,y_predict),precision_score(y_test,y_predict),recall_score(y_test,y_predict),f1_score(y_test,y_predict)
    elif task=='reg':
        reg = Ridge(alpha=1.0).fit(X_train,y_train)
        # reg = RandomForestRegressor(random_state=0).fit(X_train,y_train)
        # reg = XGBRegressor(random_state=0).fit(X_train,y_train)
        y_predict = reg.predict(X_test)
        return mean_absolute_error(y_test,y_predict),mean_squared_error(y_test,y_predict,squared=False)
    else:
        return -1