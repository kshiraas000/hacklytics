import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, ConfusionMatrixDisplay, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.ensemble import RandomForestClassifier
import time
from tabulate import tabulate
from lime.lime_tabular import LimeTabularExplainer
import torch
import torch.nn as nn
import torch.nn.functional as F

#this code follow notebook,https://www.kaggle.com/code/chandrimad31/rainfall-prediction-7-popular-models, instruction

SanDiegoPath = "San Diego.csv"
NYPath = "output.csv"

#"/Users/ringuyen/Desktop/rain forecast/dataset/NY.csv"

SanDiegoData = pd.read_csv(SanDiegoPath)
NYData = pd.read_csv(NYPath)
#fig = plt.figure(figsize = (8,5))
Yes = 0
No = 0
for i in NYData['pump']:
    if i == 1:
        Yes += 1
    else:
        No += 1

print(Yes/ (Yes + No))

#handle imbalance by oversample positive case
def classimbalance(Data, Path):
    # plot to see original data have class imbalance
    Data['pump'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in ' + Path.replace(".csv",""))
    plt.show()

    no = Data[Data['pump'] == 0]
    yes = Data[Data['pump'] == 1]
    yesOverSampled = resample(yes, replace=True, n_samples = len(no), random_state=123)
    overSample = pd.concat([no,yesOverSampled])

    #plot to see oversample solve imbalance#
    overSample['pump'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in ' + Path.replace(".csv",""))
    plt.show()
    return overSample

overSampleNY = classimbalance(NYData, NYPath)

# There is no missing data so no need to impute
# overSample.select_dtypes(include=['object']).columns
# overSample['Date'] = overSample['Date'].fillna(overSample['Date'].mode()[0])
# overSample['Location'] = overSample['Location'].fillna(overSample['Location'].mode()[0])

#label encoding as data preprocessing
def labelEncoding(data):
    lencoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        data[col] = lencoders[col].fit_transform(data[col])

labelEncoding(overSampleNY)

def removeOutlier(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print(data.shape)
    # remove outlier base on IQR
    data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(data.shape)
removeOutlier(overSampleNY)

# heatmap for corelation
def heatMapCorr(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(250, 25, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
    plt.show()
heatMapCorr(overSampleNY)

labelEncoding(NYData)
labelEncoding(NYData)
removeOutlier(NYData)


features = overSampleNY[['percent_change_1h','percent_change_24h','percent_change_7d','percent_change_30d'
                   ,'price','volume_24h','market_cap','total_supply','circulating_supply','post_active',
                   'interactions','contributors_active','contributors_created']]
target = overSampleNY['pump']

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(features, target, test_size=0.3, random_state=9)

features = NYData[['percent_change_1h','percent_change_24h','percent_change_7d','percent_change_30d'
                   ,'price','volume_24h','market_cap','total_supply','circulating_supply','post_active',
                   'interactions','contributors_active','contributors_created']]
target = NYData['pump']

X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(features, target, test_size=0.3, random_state=9)
#Normalize by MinMaxScaler
X_train_over = MinMaxScaler().fit_transform(X_train_over)
X_test_org = MinMaxScaler().fit_transform(X_test_org)

def plotRoc(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def runModel(model, X_train, y_train, X_test, y_test, verbose=True):
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plotRoc(fper, tper)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    plt.show()
    return model, accuracy, roc_auc, y_pred

# params_lr = {'penalty': 'l1', 'solver':'liblinear'}

# model_lr = LogisticRegression(**params_lr)
# model_lr, accuracy_lr, roc_auc_lr = runModel(model_lr, X_train, y_train, X_test, y_test)

# params_rf = {'max_depth': 16,
#              'min_samples_leaf': 1,
#              'min_samples_split': 2,
#              'n_estimators': 100,
#              'random_state': 12345}

# model_rf = RandomForestClassifier(**params_rf)
# model_rf, accuracy_rf, roc_auc_rf = runModel(model_rf, X_train_over,y_train_over,X_test_org,y_test_org)

params_xgb ={'n_estimators': 1000,
            'max_depth': 20}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, y_pred = runModel(model_xgb,X_train_over,y_train_over,X_test_org,y_test_org)

# class NNModel(nn.modules):
#     def __init__(self, in_features = 7, h1 = 14, h2 = 16, out_features = 1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, h1)
#         self.fc2 = nn.Linear(h1, h2)
#         self.out = nn.Softmax(h2, out_features)

#         def forward(self, x):
#             x = F.leaky_relu(self.fc1(x))
#             x = F.leaky_relu(self.fc2(x))
#             x = self.out(x)

#             return x

# model_nn = NNModel()

# X_train_tensor = torch.FloatTensor(X_train_org)
# X_test_tensor = torch.FloatTensor(X_test_org)
# Y_train_tensor = torch.LongTensor(y_train_org)
# Y_test_tensor = torch.LongTensor(y_test_org)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_nn.parameters(),lr=0.01)

# epochs = 500
# losses = []

# for i in range(epochs):
#     y_pred = model_nn.foward(X_train_tensor)
#     loss = criterion(y_pred, Y_train_tensor)

#     losses.append(loss.detach().numpy())

#     if i % 10 == 0:
#         print(f'Epoch: {i} and loss: {loss}')

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# with torch.no_grad():
#     y_eval = model_nn.forward(X_test_tensor)
#     loss = criterion(y_eval, Y_test_tensor)

# correct = 0
# with torch.no_grad():
# accuracy_scores = [accuracy_lr, accuracy_rf, accuracy_xgb]
# roc_auc_scores = [roc_auc_lr, roc_auc_rf, roc_auc_xgb]

# dataFrame = {'Model':['Logistic Regression', 'Random Forest', 'XGBoost']
#                     ,'Accuracy':accuracy_scores
#                     ,'Roc_Auc':roc_auc_scores}
# dataFrame = pd.DataFrame(dataFrame)

# sns.barplot(dataFrame, x='Model', y='Accuracy',palette='summer')
# plt.show()
# sns.barplot(dataFrame, x='Model', y='Roc_Auc',palette='summer')
# plt.show()
explainer = shap.TreeExplainer(model_xgb, X_test_org)
shap_values = explainer(X_test_org)

shap.summary_plot(shap_values,features, plot_type="bar")

# class_names = ['rain tomorrow', 'No rain tomorrow']
# feature_names = ['Date','Location', 'Temperature','Humidity','Wind Speed','Precipitation','Cloud Cover','Pressure']
# explainer = LimeTabularExplainer(X_train, feature_names =     
#                                  feature_names,
#                                  class_names = class_names, 
#                                  mode = 'classification')
# for i in range(20):
#     explaination = explainer.explain_instance(
#         data_row=X_test[i],
#         predict_fn=model_xgb.predict_proba,
#         num_features=30
#     )
# fig = explaination.as_pyplot_figure()
# plt.tight_layout()
# plt.show()