#%%
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.exceptions import DataConversionWarning

from preprocess import data_frame_min_max_norm
from preprocess import data_frame_reg_to_cls
from DQN_s_a_value import DQN1
from DQN_a_index import DQN2
from evaluation import downstream_task
from evaluation import test_task


name = "AirfoilSelfNoise"
# name = "ionosphere"
# data = pd.read_csv("AirfoilSelfNoise.csv")
data = pd.read_csv(name+".csv")
# data = pd.read_csv("ionosphere.csv")
# data = pd.read_csv("ionosphere_data.csv")
# data =data.replace({'column_ai': {'g': 1,'b':0}})

# def add_noise(data,ratio_0,ration_times):
#     ratio_len = int(data.shape[0]*data.shape[1]*ratio_0)
#     x = np.random.randint(0, data.shape[0], ratio_len)
#     y = np.random.randint(0, data.shape[1]-1, ratio_len)
#     for item in zip(x,y):
#         data.iloc[item[0],item[1]] = 8*np.random.random()
#     columns = data.columns
#     noise = pd.DataFrame(np.random.random(data.shape)*ration_times)
#     noise = pd.DataFrame(np.random.normal(0,10,size=data.shape))
#     noise = -data
#     noise.iloc[:,-1] = 0
#     data = data.to_numpy() +  noise.to_numpy()
#     return  pd.DataFrame(data,columns=columns)
# data = add_noise(data,0.9,10)
# data.to_csv('new_'+name+'.csv',index=False)






# support TASK = 'cls' METRIC = 'acc' and TASK = 'reg' METRIC = 'mae'
TASK = 'cls'
METRIC = 'acc'
# TASK = 'reg'
# METRIC = 'rae'

MIN_SIZE_TIME = 1.5
MAX_SIZE_TIME = 2.0
Original_size = data.shape[1] - 1
# normalize D0:
D0 = data_frame_min_max_norm(data)
# regression dataset to binary classification dataset
D0 = data_frame_reg_to_cls(D0)

# record the optimal  dataset and performance
D_OPT = D0
Perf_OPT,_,_,_ = downstream_task(D0, task=TASK, metric=METRIC)
# Perf_OPT = np.Inf

#ignore warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# O1 = ['sqrt', 'square','sin','cos']
O1 = ['sqrt', 'square','sin','cos','tanh']
O2 = ['+','-','*']
operation_set = O1+O2

EPISODES = 200
STEPS = 15

#the parameters of DQN1
STATE_DIM = 64
ACTION_DIM = 8
EPSILON = 0.95
MEMORY_CAPACITY=16

#the parameters of DQN2
N_STATES = STATE_DIM + ACTION_DIM
N_ACTIONS = len(operation_set)

dqn_meta = DQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
dqn_operation = DQN2(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS,MEMORY_CAPACITY=MEMORY_CAPACITY)

def Feature_DB(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.describe().iloc[i,:].describe().values)
    return feature_matrix

def choose_action(q_val_list):
    if np.random.uniform() < EPSILON:
        return np.argmax(np.array(q_val_list))
    else:
        return np.random.randint(0,len(q_val_list))

def select_meta_feature_from_data(Dg):
    Dg = Dg.drop(Dg.columns[-1],axis=1)
    state = Feature_DB(Dg)
    q_val_list = []
    for column in Dg.columns:
        feature = Dg[column]
        action = np.array(feature.describe())
        q_val_list.append(dqn_meta.get_q_value(state,action).detach().numpy()[0])
    act_index = choose_action(q_val_list)
    return Dg.iloc[:,act_index], state

def select_new_feature_from_next_Dg(Dg_new):
    Dg_new = Dg_new.drop(Dg_new.columns[-1],axis=1)
    state_ = Feature_DB(Dg_new)
    q_val_list = []
    for column in Dg_new.columns:
        feature = Dg_new[column]
        action = np.array(feature.describe())
        q_val_list.append(dqn_meta.get_q_value(state_,action).detach().numpy()[0])
    act_index = np.argmax(q_val_list)
    return Dg_new.iloc[:,act_index], state_

def select_operation_from_set(fm, Dg, operation_set):
    Dg = Dg.drop(Dg.columns[-1], axis=1)
    state = np.array(Feature_DB(Dg))
    f_state = np.array(fm.describe())
    c_state = np.hstack((state,f_state))
    operation_index = dqn_operation.choose_action(c_state)
    return operation_set[operation_index]

def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '*':
        o = np.multiply
    else:
        print("Please check your operation!")
    return o

def select_second_feature_from_set(fm,Dg,o):
    y = Dg.iloc[:,-1]
    relevance = []
    for column in Dg.columns[:-1]:
        relevance.append(normalized_mutual_info_score(o(fm,Dg[column]),y))
    feature_ind = np.argmax(relevance)
    return Dg.iloc[:,feature_ind]


# def downstream_task(Dg):
#     reg = Ridge(alpha=1.0)
#     X = Dg.iloc[:,:-1]
#     y = Dg.iloc[:,-1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=False)
#     reg.fit(X_train,y_train)
#     y_predict = reg.predict(X_test)
#     return mean_absolute_error(y_test,y_predict)

def feature_selection_from_tabular(Dg):
    pass

def cal_relevance(X,y):
  if len(X.shape) == 1:
    return normalized_mutual_info_score(X,y)
  else:
    N_col = X.shape[1]
    sum = 0
    for i in range(N_col):
      sum += normalized_mutual_info_score(X.iloc[:,i],y)
    return sum/X.shape[1]

episode = 0
perf_list = [] # list of perf, each component is a list perf_list_episode
perf_list_opt = []


acc_list = []
pre_list = []
rec_list = []
f1_list = []


while episode< EPISODES:
    perf_list_episode = [] #each component is a step perf
    perf_list_episode_opt = []
    Dg = D0
    step = 0
    while step < STEPS:
        #select meta feature by RL
        fm, state_meta = select_meta_feature_from_data(Dg)
        #select operation from operations set by RL
        o = select_operation_from_set(fm, Dg, operation_set)
        #justify type of the selected operation and generate features.
        fname = ''
        fg = fm
        o_index = operation_set.index(o)
        if o in O1:
            fname = fm.name + '_' + o
            o = justify_operation_type(o)
            fg = o(fm)
        if o in O2:
            #select the second feature from the remaining feature set.
            # fname = fm.name + '_' + o
            fname = fm.name + o
            o = justify_operation_type(o)
            fm2 = select_second_feature_from_set(fm,Dg,o)
            fg = o(fm,fm2)
            fname = fname +fm2.name
        # if the generated feature name already exist in the dataset, continue
        if fname in Dg.columns:
            continue
        # if the generated feature value already exist in the dataset, continue
        for column in Dg.columns:
            if max(np.abs(fg  - np.array(Dg.loc[:,column]))) < 1e-10:
                continue
        # normalize fg
        scaler = MinMaxScaler()
        fg = scaler.fit_transform(np.expand_dims(fg,axis=1)).squeeze()
        #add fg into the Dg
        Dg_new = Dg.copy()
        Dg_new.insert(len(Dg.columns)-1,fname,np.array(fg))
        fm_, state_meta_ = select_new_feature_from_next_Dg(Dg_new)
        # Dg_new = pd.concat([Dg,pd.DataFrame(fg)],axis=1)
        new_perf, prec_score, rec_score, f1_score = downstream_task(Dg_new, task=TASK, metric=METRIC)

        acc_list.append(new_perf)
        pre_list.append(prec_score)
        rec_list.append(rec_score)
        f1_list.append(f1_score)


        old_perf,_,_,_ = downstream_task(Dg, task=TASK, metric=METRIC)



        reward_meta = new_perf - old_perf
        dqn_meta.store_transition(state_meta, np.array(fm.describe()),reward_meta, state_meta_, np.array(fm_.describe()))
        # update dqn_meta,
        if dqn_meta.memory_counter > dqn_meta.MEMORY_CAPACITY:
            dqn_meta.learn()

        #update dqn_operation
        state_operation = np.hstack((state_meta,fm.describe()))
        state_operation_ = np.hstack((state_meta_, fm_.describe()))
        y = Dg.iloc[:,-1]
        rel_Dg = cal_relevance(Dg,y)
        # rel_Dg_new = cal_relevance(Dg_new)
        # reward = rel_Dg_new - rel_Dg
        reward = 1/Dg_new.shape[1] *(normalized_mutual_info_score(fg,y) - 1/Dg.shape[1]*rel_Dg)
        dqn_operation.store_transition(state_operation, o_index, reward, state_operation)
        if dqn_operation.memory_counter > dqn_operation.MEMORY_CAPACITY:
            dqn_operation.learn()

        # print(new_perf)
        Dg = Dg_new




        if new_perf > Perf_OPT:
            # Dg = Dg_new
            D_OPT  = Dg_new
            Perf_OPT = new_perf

        #conduct feature selection to control the data size of Dg
        # Dg = feature_selection_from_tabular(Dg)
        if Dg.shape[1] - 1 >= int(Original_size*MAX_SIZE_TIME):
            X, y = Dg.iloc[:,:-1], Dg.iloc[:,-1]
            fitted = SelectKBest(mutual_info_regression, k=int(Original_size*MIN_SIZE_TIME)).fit(X, y)
            X_new = X.loc[:,fitted.get_support()]
            Dg = pd.concat([X_new, y], axis=1)


        print("New performance is: {:.3f}, Best performance is: {:.3f}".format(new_perf, Perf_OPT))
        print("Step {} ends!".format(step))
        # record perf
        perf_list_episode.append(new_perf)
        perf_list_episode_opt.append(Perf_OPT)
        step += 1
    print("Episode {} ends!".format(episode))
    perf_list.append(perf_list_episode)
    perf_list_opt.append(perf_list_episode_opt)
    episode += 1

acc_list = pd.DataFrame(acc_list)
pre_list = pd.DataFrame(pre_list)
rec_list = pd.DataFrame(rec_list)
f1_list = pd.DataFrame(f1_list)

acc_list.to_csv("accuracy.csv",header=False,index=False)
pre_list.to_csv("precision.csv",header=False,index=False)
rec_list.to_csv("recall.csv",header=False,index=False)
f1_list.to_csv("f1.csv",header=False,index=False)
print("data collection is ended!")


perf_df = pd.DataFrame(perf_list).T
perf_opt_df = pd.DataFrame(perf_list_opt).T
# perf_df.to_csv("perf.csv")
# perf_opt_df.to_csv("perf_opt.csv")
Dg.to_csv("generated.csv")

# Evaluation:
if TASK == 'reg':
    mae0, rmse0, rae0 = test_task(D0,task=TASK)
    mae1, rmse1, rae1 = test_task(D_OPT,task=TASK)
    # print(mae0, mae1, rmse0, rmse1)
    print("MAE on original is: {:.3f}, MAE on generated is: {:.3f}".format(mae0, mae1))
    print("RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}".format(rmse0, rmse1))
    print("1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}".format(1-rae0, 1-rae1))
if TASK == 'cls':
    acc0, precision0, recall0,f1_0 = test_task(D0,task=TASK)
    acc1, precision1, recall1,f1_1 = test_task(D_OPT,task=TASK)
    # print(acc0, acc1, precision0,precision1, recall0,recall1,f1_0,f1_1)
    print("Acc on original is: {:.3f}, Acc on generated is: {:.3f}".format(acc0, acc1))
    print("Pre on original is: {:.3f}, Pre on generated is: {:.3f}".format(precision0, precision1))
    print("Rec on original is: {:.3f}, Rec on generated is: {:.3f}".format(recall0, recall1))
    print("F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}".format(f1_0, f1_1))