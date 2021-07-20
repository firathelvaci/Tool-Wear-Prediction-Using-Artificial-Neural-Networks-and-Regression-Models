import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.metrics import binary_accuracy
from sklearn.metrics import f1_score, roc_auc_score,roc_curve,auc
#read data
main_df=pd.read_csv('../input/tool-wear-detection-in-cnc-mill/train.csv')
main_df=main_df.fillna('no')

files = list()

for i in range(1,19):
    exp_number = '0' + str(i) if i < 10 else str(i)
    file = pd.read_csv("../input/tool-wear-detection-in-cnc-mill/experiment_{}.csv".format(exp_number))
    row = main_df[main_df['No'] == i]
    
     #add experiment settings to features
    file['feedrate']=row.iloc[0]['feedrate']
    file['clamp_pressure']=row.iloc[0]['clamp_pressure']
    file=file.drop(['Machining_Process'],axis=1)
    # Having label as 'tool_conidtion'
    file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    files.append(file)
df = pd.concat(files, ignore_index = True)
print(df.info())
#Sampling rate 100ms
for i in range(0,17):
    time= np.arange(0,len(files[i])/10,0.1)
    plt.style.use('ggplot')
    plt.plot(time,files[i]['X1_ActualPosition'], label = 'X1_ActualPosition')
    plt.plot(time,files[i]['Y1_ActualPosition'].values, label = 'Y1_ActualPosition')
    plt.plot(time,files[i]['Z1_ActualPosition'].values, label = 'Z1_ActualPosition')
    plt.gca().set_aspect('auto', adjustable='datalim')
    plt.title("Actual Position Comparison over Time")
    plt.xlabel("time (ms)")
    plt.ylabel("Actual Position (mm)")
    plt.legend()
    plt.show()
    print(i+1)
plt.plot(files[3]['X1_ActualPosition'],files[3]['Y1_ActualPosition'].values, label = 'X Y location of tool')
plt.title("X Y location of tool")
plt.xlabel("X Actual Position mm")
plt.ylabel("Y Actual Position mm")
plt.legend()
plt.show()


frames=['X1_ActualPosition','X1_ActualVelocity','X1_ActualAcceleration','X1_CommandPosition',
        'X1_CommandVelocity','X1_CommandAcceleration','X1_CurrentFeedback','X1_DCBusVoltage',
        'X1_OutputCurrent','X1_OutputVoltage','X1_OutputPower','Y1_ActualPosition','Y1_ActualVelocity',
        'Y1_ActualAcceleration','Y1_CommandPosition','Y1_CommandVelocity','Y1_CommandAcceleration',
        'Y1_CurrentFeedback','Y1_DCBusVoltage','Y1_OutputCurrent','Y1_OutputVoltage','Y1_OutputPower',
        'Z1_ActualPosition','Z1_ActualVelocity','Z1_ActualAcceleration','Z1_CommandPosition','Z1_CommandVelocity',
        'Z1_CommandAcceleration','Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage',
        'S1_ActualPosition','S1_ActualVelocity','S1_ActualAcceleration','S1_CommandPosition','S1_CommandVelocity',
        'S1_CommandAcceleration','S1_CurrentFeedback','S1_DCBusVoltage','S1_OutputCurrent','S1_OutputVoltage','S1_OutputPower',
        'S1_SystemInertia','M1_CURRENT_PROGRAM_NUMBER','M1_sequence_number','M1_CURRENT_FEEDRATE'
        ,'feedrate','clamp_pressure']


for i in range(18):
    for frame in frames:
        Q1 = files[i][frame].quantile(0.25)
        Q3 = files[i][frame].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        print(frame)
        print(upper_limit)
        print(lower_limit)
        median=files[i][frame].median()
        #files[i].drop(files[i][ (files[i][frame] > upper_limit) | (files[i][frame] < lower_limit) ].index , inplace=True)
        files[i].at[files[i][(files[i][frame] > upper_limit) | (files[i][frame] < lower_limit) ].index, [frame]]=median

for i in range(0,17):
 time= np.arange(0,len(files[i])/10,0.1)

 plt.style.use('ggplot')
 plt.plot(time,files[i]['X1_ActualPosition'], label = 'X1_ActualPosition')
 plt.plot(time,files[i]['Y1_ActualPosition'].values, label = 'Y1_ActualPosition')
 plt.plot(time,files[i]['Z1_ActualPosition'].values, label = 'Z1_ActualPosition')
 plt.gca().set_aspect('auto', adjustable='datalim')
 plt.title("Actual Position Comparison over Time experiment")
 plt.xlabel("time (ms)")
 plt.ylabel("Actual Position (mm)")
 plt.legend()
 plt.show()
 print(i+1)

plt.plot(files[1]['X1_ActualPosition'],files[1]['Y1_ActualPosition'].values, label = 'X Y location of tool')
plt.title("X Y location of tool")
plt.xlabel("X Actual Position mm")
plt.ylabel("Y Actual Position mm")
plt.legend()
plt.show()

df = pd.concat(files, ignore_index = True)
df.describe()
X=df.drop(['label'],axis=1).values
Y=df['label'].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,  test_size=0.3,random_state=0)

print(X_Train.shape)
print(X_Test.shape)
df.describe()
df['S1_ActualVelocity'].values

Neuralmodel = keras.Sequential()
Neuralmodel.add(keras.Input(shape=(49)))
Neuralmodel.add(layers.Dense(512, activation="tanh"))
Neuralmodel.add(layers.Dropout(0.5))
Neuralmodel.add(layers.BatchNormalization())
Neuralmodel.add(layers.Dropout(0.4))
Neuralmodel.add(layers.Dense(512/2,activation="tanh"))
Neuralmodel.add(layers.BatchNormalization())
Neuralmodel.add(layers.Dropout(0.3))
Neuralmodel.add(layers.Dense(512/8,activation="tanh"))
Neuralmodel.add(layers.BatchNormalization())
Neuralmodel.add(layers.Dropout(0.2))
Neuralmodel.add(layers.Dense(32,activation="relu"))
Neuralmodel.add(layers.BatchNormalization())
Neuralmodel.add(layers.Dense(1, activation="sigmoid" ))


keras.optimizers.Adam(lr=0.0001)
# Compile model
Neuralmodel.compile(loss='binary_crossentropy', optimizer='adam',metrics=['binary_accuracy'])
Neuralmodel.summary()
# Fit the model
Neuralhist =Neuralmodel.fit(X_Train, Y_Train,validation_data=(X_Test,Y_Test), batch_size=100000, epochs=3000)


plt.style.use('ggplot')
plt.plot(Neuralhist.history['loss'], label = 'loss')
plt.plot(Neuralhist.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.style.use('ggplot')
plt.plot(Neuralhist.history['binary_accuracy'], label = 'binary_accuracy')
plt.plot(Neuralhist.history['val_binary_accuracy'], label='val_binary_accuracy')
plt.title("binary_accuracy vs val_binary_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

NeuralScorePra=Neuralmodel.predict(X_Test)
NeuralScore=np.zeros(len(Y_Test))

for j in range(len(Y_Test)):
    if NeuralScorePra[j]>0.6:
        NeuralScore[j]=1
    else:
        NeuralScore[j]=0

print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, NeuralScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, NeuralScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, NeuralScore)))
    
    
DTmodel = DecisionTreeClassifier(criterion='gini', random_state=42)
DTmodel.fit(X_Train, Y_Train)
DTScore = DTmodel.predict(X_Test)

print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, DTScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, DTScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, DTScore)))

RFmodel = RandomForestClassifier(max_depth=5, n_estimators=10)
RFmodel.fit(X_Train, Y_Train)
RFScore = RFmodel.predict(X_Test)


print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, RFScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, RFScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, RFScore)))

# Support vector regression
from sklearn.svm import SVC
SVCModel = SVC(probability=True)
SVCModel.fit(X_Train,Y_Train)
SVCScore = SVCModel.predict(X_Test)
print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, SVCScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, SVCScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, SVCScore)))

from sklearn.linear_model import LogisticRegression
log_regModel=LogisticRegression()
log_regModel.fit(X_Train,Y_Train)
log_regScore = log_regModel.predict(X_Test)
print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, log_regScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, log_regScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, log_regScore)))

from sklearn.neighbors import KNeighborsClassifier
knbModel=KNeighborsClassifier()
knbModel.fit(X_Train,Y_Train)
knbScore = knbModel.predict(X_Test)
print("Accuracy: {0:0.4f}".format(binary_accuracy(Y_Test, knbScore)))
print("F1 Score: {0:0.4f}".format(f1_score(Y_Test, knbScore)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(Y_Test, knbScore)))

NeuralScore=Neuralmodel.predict(X_Test)
DTScore = DTmodel.predict_proba(X_Test)[:,1]
RFScore = RFmodel.predict_proba(X_Test)[:,1]
SVCScore = SVCModel.predict_proba(X_Test)[:,1]
RFScore = RFmodel.predict_proba(X_Test)[:,1]
knbScore= knbModel.predict_proba(X_Test)[::,1]
log_regScore= log_regModel.predict(X_Test)
print(knbScore)

fpr_Neural, tpr_Neural, thresholds_Neural = roc_curve(Y_Test, NeuralScore)
auc_Neural = auc(fpr_Neural, tpr_Neural)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(Y_Test, DTScore)
auc_DT= auc(fpr_DT, tpr_DT)
fpr_RF, tpr_RF, thresholds_RF= roc_curve(Y_Test, RFScore)
auc_RF= auc(fpr_RF, tpr_RF)
fpr_SVC, tpr_SVC, thresholds_SVC= roc_curve(Y_Test, SVCScore)
auc_SVC= auc(fpr_SVC, tpr_SVC)

fpr_knb, tpr_knb, thresholds_knb= roc_curve(Y_Test, knbScore)
auc_knb= auc(fpr_knb, tpr_knb)

fpr_log_reg, tpr_log_reg, thresholds_log_reg= roc_curve(Y_Test, log_regScore)
auc_log_reg= auc(fpr_log_reg, tpr_log_reg)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Neural, tpr_Neural, label='Neural Network (area = {:.3f})'.format(auc_Neural))
plt.plot(fpr_DT, tpr_DT, label='DT (area = {:.3f})'.format(auc_DT))
plt.plot(fpr_RF, tpr_RF, label='RF (area = {:.3f})'.format(auc_RF))

plt.plot(fpr_SVC, tpr_SVC, label='SVC (area = {:.3f})'.format(auc_SVC))
plt.plot(fpr_log_reg, tpr_log_reg, label='log_reg (area = {:.3f})'.format(auc_log_reg))
plt.plot(fpr_knb, tpr_knb, label='knb (area = {:.3f})'.format(auc_knb))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()