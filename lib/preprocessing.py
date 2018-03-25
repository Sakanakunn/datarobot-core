import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import pdb
from sklearn.metrics import roc_curve, auc
import subprocess

plt.style.use('ggplot')
font = {'family' : 'meiryo'}
plt.rc('font', **font)

## スコアリング用データのImport
def train_read(CsvPath,score_column_name,Exclude_columns):
    df = pd.read_csv(CsvPath, header=0)
    train_keys = ['X_train','y_train','all_train_columns','objects_train_columns','objects_dummy_train_columns','dtype_dict']
    if Exclude_columns != ['']:
        X_train = df.iloc[:, df.columns != score_column_name].drop(Exclude_columns, axis=1)
    else:
        X_train = df.iloc[:, :train_num]
    train_values = [
    ## 特徴量カラムの読み込み
    X_train
    ## 正解ラベルカラムの読み込み
    ,df.iloc[:,df.columns == score_column_name]
    ## 全カラム名のArray
    ,df.columns.values
    ## Objects型に絞り込んだカラム名のArray
    ,df.loc[:,extract_objects_columns(df)].columns
    ## 正解ラベルカラムの読み込み
    ,X_train.loc[:,extract_objects_columns(X_train)].columns
    ## データ型を生成
    ,build_dtype_dict(df.loc[:, extract_objects_columns(df)].columns)
    ]
    ## return を dict型で生成
    train_set = dict(zip(train_keys,train_values))
    return train_set

## スコアリング用データのImport
def score_read(CsvPath,Exclude_columns,dtype_dict):
    df = pd.read_csv(CsvPath,dtype = dtype_dict,header=0)
    score_keys = ['X_score']

    if Exclude_columns != ['']:
        X_score = df.drop(Exclude_columns, axis=1)
    else:
        X_score = df
    score_values = [
    ## 特徴量カラムの読み込み
    X_score
    ]
    ## return を dict型で生成
    score_set = dict(zip(score_keys,score_values))
    return score_set


def extract_objects_columns(df):
    dummy_columns_indexes = []
    for i in range(0, df.shape[1]):
        if df.dtypes[i].str == '|O':
            dummy_columns_indexes.append(True)
        else:
            dummy_columns_indexes.append(False)
    return dummy_columns_indexes

def build_dtype_dict(obj_columns):
    values_dict = []
    for i in range(0, obj_columns.values.size):
        values_dict.append(object)
    print(values_dict)
    dtype_dict = dict(zip(obj_columns.values,values_dict))
    return dtype_dict

def onehot_encode(df,object_columns):
    df_ohe = pd.get_dummies(
        df
        ,dummy_na = True
        ,columns = object_columns
    )
    onehot_set = [
        df_ohe
        ,df_ohe.columns
    ]
    return  onehot_set


def impute_missingvalue(df,dummy_columns):
    imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
    imp.fit(df)
    imputed_df = pd.DataFrame(imp.transform(df),columns = dummy_columns )
    return imputed_df


def feature_selection_rfe(df,y):
    selector = RFE(GradientBoostingClassifier(random_state=1)
                   ,n_features_to_select = 10
                   ,step = .005)
    selector.fit(df, y.as_matrix().ravel())
    df_fin = pd.DataFrame(selector.transform(df)
                          ,columns = df.columns[selector.support_])
    return df_fin

def integrate_columns(train_columns,score_df):
    cols_model = set(train_columns)
    cols_score = set(score_df.columns)
    diff1 = cols_model - cols_score
    diff2 = cols_score - cols_model
    df_cols_m = pd.DataFrame(None,columns = train_columns,dtype = float)
    df_cols_m_conc = pd.concat([df_cols_m,score_df])
    df_cols_m_conc_ex = df_cols_m_conc.drop(list(diff2),axis = 1)
    df_cols_m_conc_ex.loc[:, list(diff1)] = df_cols_m_conc_ex.loc[:, list(diff1)].fillna(0, axis=1)
    df_cols_m_conc_ex_reorder = df_cols_m_conc_ex.reindex(train_columns, axis=1)
    return df_cols_m_conc_ex_reorder

def x_check(df_tr,df_sc):
    print(df_tr.info())
    print(df_sc.info())
    print(df_tr.head())
    print(df_sc.head())
    print(set(df_tr.columns.values) - set(df_sc.columns.values))

def build_pipeline(classifiers,classifier_pipe_names):
    pipelines = []
    for i in classifiers:
        pipelines.append(Pipeline([('scl',StandardScaler()),('est',i)]))
    pipelines_dict = dict(zip(classifier_pipe_names,pipelines))
    return pipelines_dict

def train_pipeline_with_grid(pipeline_dict,X_train,y_train):
    # パラメータグリッドの設定
    grid_parameters = []
    grid_parameters.append({'est__C': [1,50,100], 'est__penalty': ['l1', 'l2']})
    grid_parameters.append({'est__n_estimators':[1,50,100]
                            ,'est__criterion': ['gini','entropy']})
    grid_parameters.append({'est__n_estimators': [1,50,100]
                            ,'est__subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]})
    grid_parameters.append({'est__learning_rate': [0.1, 0.3, 0.5],
              'est__max_depth': [2, 3, 5, 10],
              'est__subsample': [0.5, 0.8, 0.9, 1]
              })
    trained_pipeline_dict = {}

    for i,j in zip(pipeline_dict.keys(),grid_parameters):
        print(y_train)
        print(pipeline_dict[i])
        print(j)
        gs = GridSearchCV(estimator=pipeline_dict[i], param_grid=j, scoring='f1', cv=3, n_jobs = -1)
        trained_pipeline_dict[i] = gs.fit(X_train,y_train.as_matrix().ravel())
    return trained_pipeline_dict

def split_holdout(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train,y_train,X_test,y_test

def Scoring_TrainedModel(trained_pipeline_dict,X_test,y_test):
    alg_names = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    roc_scores = []
    f1_scores = []
    result_dict = {}
    for key in trained_pipeline_dict.keys():
        alg_names.append(key)
        accuracy_scores.append(float(accuracy_score(y_test.as_matrix().ravel(),trained_pipeline_dict[key].predict(X_test))))
        precision_scores.append(float(precision_score(y_test.as_matrix().ravel(), trained_pipeline_dict[key].predict(X_test))))
        recall_scores.append(float(recall_score(y_test.as_matrix().ravel(), trained_pipeline_dict[key].predict(X_test))))
        f1_scores.append(float(f1_score(y_test.as_matrix().ravel(), trained_pipeline_dict[key].predict(X_test))))
        fpr, tpr, thresholds = metrics.roc_curve(y_test.as_matrix().ravel(), trained_pipeline_dict[key].predict(X_test))
        roc_scores.append(metrics.auc(fpr, tpr))
    result_dict['0_alg_name'] = alg_names
    result_dict['1_accuracy_score'] = accuracy_scores
    result_dict['2_precision_score'] = precision_scores
    result_dict['3_recall_score'] = recall_scores
    result_dict['4_auc_score'] = roc_scores
    result_dict['5_F1_score'] = f1_scores
    result_dict = pd.DataFrame.from_dict(result_dict)
    result_dict.to_csv('result.csv')
    plot = result_dict.plot.barh(figsize=(15,10),subplots = True, x = '0_alg_name' ,layout = (2,3))
    fig = plot[0][0].get_figure()
    fig.savefig("result_fig_0.png")

    #変数の重要度を計算する
    val_imp = ExtraTreesClassifier()
    val_imp.fit(X_test,y_test.as_matrix().ravel())
    val_imp_values = []
    for i in val_imp.feature_importances_:
        array = []
        array.append(round(i,2))
        val_imp_values.append(array)

    imp_dict = dict(zip(X_test.columns.values,val_imp_values))
    result_dict_imp = pd.DataFrame.from_dict(imp_dict)
    result_dict_imp_fin = result_dict_imp.T.rename(columns = {0:'Feature Importance'}).sort_values(by = 'Feature Importance', ascending = False)
    result_dict_imp_fin.to_csv('imp.csv')
    plot = result_dict_imp_fin.plot.bar(y="Feature Importance",figsize = (10,10))
    fig = plot.get_figure()
    fig.savefig("importance_fig.png")

    # Partial Dependence
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_test,y_test.as_matrix().ravel())
    fig, axs = plot_partial_dependence(gb_model,
                                       features = result_dict_imp_fin.iloc[0:3,0].index.values,
                                       feature_names= result_dict_imp_fin.iloc[0:3,0].index.values,
                                       X=X_test,
                                       grid_resolution=5)
    fig.savefig("pdp_plot.png")
    return result_dict


def add_prediction_scores(trained_pipeline_dict,X_score):
    predict_scores = []
    predict_scores_keys = ['Log_pred','RF_pred','GB_pred','Xgb_pred']
    for key in trained_pipeline_dict.keys():
        TF_prob = trained_pipeline_dict[key].predict_proba(X_score)
        predict_scores_sub = []
        for val in TF_prob:
            predict_scores_sub.append(round(val[0],4))
        predict_scores.append(predict_scores_sub)
    predict_dict = dict(zip(predict_scores_keys,predict_scores))
    predict_df = pd.DataFrame.from_dict(predict_dict)
    predict_df_conc = pd.concat([X_score, predict_df], axis=1)
    predict_df_conc.to_csv('output.csv')
    convert_df_to_image(predict_df_conc, 'result.png', False)
    pdb.set_trace()
    return  predict_df_conc

def convert_df_to_image(df,imagename,all_fl):
    dphtml = r'<link rel="stylesheet" type="text/css" href="./table.css" />' + '\n'
    if all_fl:
        dphtml += df.to_html()
    else:
        dphtml += df.head().to_html()

    with open('table.html', 'w') as f:
        f.write(dphtml)
        f.close()
        pass
    subprocess.call(
        'wkhtmltoimage -f png --width 0 table.html '+ imagename, shell=True)

## TODO ベストモデルを選択する!
## TODO ベストモデルを保存する!
## TODO 予測ラベルを評価用データセットに付与する！
