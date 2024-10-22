import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import optuna
import datetime
import matplotlib.pyplot as plt

xgb_boost= xgb.XGBRegressor(device = 'cuda:0',
                            tree_method='hist',
                            min_child_weight=45,
                            sampling_method='gradient_based',
                            num_parallel_tree=4,
                            colsample_bylevel=0.8,
                            colsample_bynode=0.7,
                            booster='gbtree',
                            grow_policy='depthwise',
                            subsample=0.6,
                            objective ='reg:squarederror', 
                            colsample_bytree = 0.65,
                            eta=0.35, 
                            learning_rate = 0.1,
                            max_depth = 6, 
                            reg_alpha=5.5, 
                            n_estimators = 4000,
                            reg_lambda=3.5)


def get_house_year(row):
    if row['yr_renovated'] == 0:
        return row['sale_yr'] - row['yr_built']
    else:
        return row['sale_yr'] - row['yr_renovated']

def data_dreprocessing(df_train,df_test):
    # train data
    # remove non-useful columns
    non_useful_columns = ['id','zipcode','sqft_lot15']
    df_train = df_train.drop(non_useful_columns,axis=1)
    # feature engineering
    df_train['is_basement'] = df_train['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
    df_train['house_year'] = df_train.apply(get_house_year,axis=1)
    # feature selection (remove low correlation with target, threshold = 0.2)
    corr = df_train.corr(numeric_only=True).abs().sort_values(by='price', ascending=False)
    select_by_corr = corr[(corr["price"] >= 0.2)].index
    var = VarianceThreshold(threshold=25)
    var.fit(df_train)
    select_by_var = df_train.columns[var.get_support()]

    # Create a SelectKBest object and fit it to the data using f_regression
    kbest_selector = SelectKBest(score_func=f_regression, k=10)
    selected_features_kbest = kbest_selector.fit_transform(df_train.drop('price', axis=1),df_train['price'].copy())
    selected_feature_indices_kbest = kbest_selector.get_support(indices=True)
    selectKbest_df = df_train.iloc[:, selected_feature_indices_kbest].copy()
    select_by_Kbest = df_train.iloc[:, selected_feature_indices_kbest].columns
    feature=list(set(list(select_by_corr) + list(select_by_var) + list(select_by_Kbest)))
    feature.remove('price')
    # test data
    # remove non-useful columns
    df_test = df_test.drop(non_useful_columns,axis=1)
    # feature engineering
    df_test['is_basement'] = df_test['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
    df_test['house_year'] = df_test.apply(get_house_year,axis=1)
    
    # concat train and validation data as big data
    y_train = df_train['price']
    x_train = df_train.drop('price',axis=1)[feature]
    # keep last 20% of training data as validation set
    y_vaild = y_train[int(0.8*len(y_train)):]
    x_vaild = x_train[int(0.8*len(x_train)):][feature]
    # # keep first 80% of training data as training set
    y_train = y_train[:int(0.8*len(y_train))]
    x_train = x_train[:int(0.8*len(x_train))][feature]
    # test data
    x_test = df_test[feature]
    return x_train, y_train, x_vaild, y_vaild, x_test


def train(x_train, y_train, x_vaild, y_vaild, save_path=None):
    model = xgb_boost
    # Create empty lists to store training and validation errors
    train_errors = []
    valid_errors = []

    # fit model and save learning curve
    eval_set = [(x_train, y_train), (x_vaild, y_vaild)]
    model.fit(x_train, y_train, eval_metric=["mae"], eval_set=eval_set, verbose=True)

    # Get the training and validation errors from the trained model
    results = model.evals_result()
    train_errors = results['validation_0']['mae']
    valid_errors = results['validation_1']['mae']

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error')
    plt.plot(range(1, len(valid_errors) + 1), valid_errors, label='Validation Error')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('XGBoost Learning Curve')
    
    # Save the learning curve image if save_path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    return model


def valid(model, x_vaild, y_vaild):
    preds = model.predict(x_vaild)
    rmse = mean_absolute_error(y_vaild, preds)
    return rmse

def get_submission(model, x_test, path):
    preds = model.predict(x_test)
    submission = pd.DataFrame({'id':range(1,len(preds)+1),'price':preds})
    submission.to_csv(path, index=False)

def time_print(message):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)