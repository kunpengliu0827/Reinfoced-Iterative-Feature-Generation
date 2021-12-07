import pandas as pd
import numpy as np
from sklearn import preprocessing

def data_frame_min_max_norm(D):
    scaler = preprocessing.MinMaxScaler()
    columns = D.columns
    D_norm_value = pd.DataFrame(scaler.fit_transform(D[columns[:-1]]))
    D_norm_value.columns = columns[:-1]
    D_norm = pd.concat([D_norm_value,D[columns[-1]]],axis=1)
    return D_norm

# X features can be categorized into 4 groups: textual categorical, numerical categorical, continuous and binary.
# textual categorical: {'Monday','Tuesday',...}
# numerical categorical: {'0', '1', '2'}, where '0' denotes 'Monday', '1'  denotes 'Tuesday'
# continuous: {3.12, 2.53,...}; binary: {1,0}. In Python, binary can be viewed as continuous
# columns_cat_tex = ['nationality','club','preferred_foot','work_rate']
# transform textual categorical variable into numerical categorical variable
def cat_tex_to_cat_num(data):
    encoder = preprocessing.LabelEncoder()
    for var in data.columns:
        data.loc[:,var] = encoder.fit_transform(data.loc[:,var])
    return pd.DataFrame(data)
# transform numerical categorical variable into one-hot variable
def cat_num_to_oh(data):
    encoder = preprocessing.OneHotEncoder()
    data_oh = encoder.fit_transform(data).toarray()
    return pd.DataFrame(data_oh)
def data_frame_one_hot_encode(D,columns_cat_tex, columns_cat_num, columns_continuous,column_label):
    X_cat_tex = D.loc[:, columns_cat_tex]
    X_cat_num = D.loc[:, columns_cat_num]
    X_continuous = D.loc[:, columns_continuous]
    y = D.loc[:, column_label]
    # transform categorical variables into one-hot variables
    X_cat_num_from_tex = cat_tex_to_cat_num(X_cat_tex)
    X_cat_num_joint = pd.concat([X_cat_num_from_tex, X_cat_num], axis=1)
    X_one_hot = cat_num_to_oh(X_cat_num_joint)
    X = pd.concat([X_continuous, X_one_hot], axis=1)

    D_encoded = pd.concat([X,y], axis = 1)
    return D_encoded
def data_frame_reg_to_cls(D):
    X = D.iloc[:,:-1]
    y_name = D.columns[-1]
    y =D.iloc[:,-1]>=D.iloc[:,-1].median()
    # y = D.iloc[:, -1] >= D.iloc[:, -1].mean()
    y_series = pd.Series(y,name=y_name)
    D_cls = pd.concat([X,y_series],axis=1)
    return  D_cls