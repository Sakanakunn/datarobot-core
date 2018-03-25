import sys
sys.path.append("./lib/")
import preprocessing as pre

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pdb

def main(train_data_path,score_column_name,mapping_fl,exclude_columns_names):
    ## 訓練データの前処理
    ### データの読み込み
    train_set = pre.train_read(train_data_path,score_column_name,exclude_columns_names)

    print(train_set.keys())

    ### one-hot Encoding
    x_train_ohe = pre.onehot_encode(
        train_set['X_train']
        ,train_set['objects_dummy_train_columns']
    )

    ### Impute missing values
    x_train_ohe_imp = pre.impute_missingvalue(
        x_train_ohe[0]
        ,x_train_ohe[1]
    )

    ### Feature Selection with RFE
    x_train_fin = pre.feature_selection_rfe(x_train_ohe_imp,train_set["y_train"])

    print("-------------------------------")
    print(x_train_fin.shape)
    print(x_train_fin.head())
    print(x_train_fin.info())
    print("-------------------------------")

    ## スコアリング用データの前処理
    ### データの読み込み
    score_set = pre.score_read(score_data_path,exclude_columns_names,train_set['dtype_dict'])


    ### one-hot Encoding
    x_score_ohe = pre.onehot_encode(
        score_set['X_score']
        ,train_set['objects_dummy_train_columns']
    )

    ## カラム構成の統一 & reorder
    x_score_ohe_inte = pre.integrate_columns(x_train_ohe[1],x_score_ohe[0])

    ## Impute
    x_score_ohe_inte_imp = pre.impute_missingvalue(
        x_score_ohe_inte
        ,x_train_ohe[1]
    )

    ## カラム選択
    x_score_fin = x_score_ohe_inte_imp.loc[:,x_train_fin.columns.values]

    pre.x_check(x_train_fin,x_score_fin)

    ## パイプラインの用意
    classifiers = [
        LogisticRegression(random_state = 10)
        ,RandomForestClassifier(random_state = 10)
        ,GradientBoostingClassifier(random_state = 10)
        ,xgb.XGBClassifier()
    ]

    classifier_pipe_names =[
        'Logistic_pipe'
        ,'RandomForest_pipe'
        ,'GradientBoost_pipe'
        ,'XgbRegression'
    ]

    if mapping_fl != False:
        # ローン審査でNOとなったサンプルを1（正例）として変換
        class_mapping = {'N':1, 'Y':0}
        train_set['y_train'] = train_set['y_train'].map(class_mapping)

    pipelines_dict = pre.build_pipeline(classifiers,classifier_pipe_names)

    X_train,y_train,X_test,y_test = pre.split_holdout(x_train_fin,train_set['y_train'])
    print(y_train)

    trained_pipeline_dict = pre.train_pipeline_with_grid(pipelines_dict,X_train,y_train)

    result_dict = pre.Scoring_TrainedModel(trained_pipeline_dict,X_test,y_test)
    print(result_dict)

if __name__ == '__main__':
    ## テストデータのパス
    train_data_path = './data/titanic/train.csv'
    ##Class Mapping
    mapping_fl = False
    ## スコアリング列のカラム番号
    scoring_column_name = 'Survived'
    ## 除外するカラム名
    exclude_columns_names = ['Ticket','Cabin','Name','PassengerId']
    ## スコアリングデータのパス
    score_data_path =  './data/titanic/test.csv'

    main(train_data_path,scoring_column_name,mapping_fl,exclude_columns_names)