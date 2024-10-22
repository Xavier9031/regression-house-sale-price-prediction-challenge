import pandas as pd
import numpy as np
import os
import argparse
import time
from models import *

import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# example: python main.py --get_submission --submission_path submission.csv
argparser = argparse.ArgumentParser()
argparser.add_argument('--get_submission', action='store_true')
argparser.add_argument('--submission_path', type=str, default='submission.csv')



# Data Path
path_train = os.path.join(os.getcwd(), "data", "train-v3.csv")
path_valid = os.path.join(os.getcwd(), "data", "valid-v3.csv")
path_test = os.path.join(os.getcwd(), "data", "test-v3.csv")



if __name__ == '__main__':
    args = argparser.parse_args()
    # Load data
    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df_test = pd.read_csv(path_test)
    df_train_all = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)

    # Preprocessing
    time_print("Preprocessing start")
    x_train, y_train, x_vaild, y_vaild, x_test = data_dreprocessing(df_train_all, df_test)
    time_print("Preprocessing finished")

    # Tune hyperparameters
    # time_print("Tuning start")
    # def objective(trial):
    #     params = {
    #         'device':'cuda:0',
    #         'tree_method':'hist',
    #         'min_child_weight':trial.suggest_int('min_child_weight', 1, 50),
    #         'sampling_method':trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),
    #         'num_parallel_tree':trial.suggest_int('num_parallel_tree', 1, 10),
    #         'colsample_bylevel':trial.suggest_float ('colsample_bylevel', 0.5, 1),
    #         'colsample_bynode':trial.suggest_float ('colsample_bynode', 0.5, 1),
    #         'booster':trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #         'grow_policy':trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    #         'subsample':trial.suggest_float ('subsample', 0.5, 1),
    #         'objective':'reg:squarederror',
    #         'colsample_bytree':trial.suggest_float ('colsample_bytree', 0.5, 1),
    #         'eta':trial.suggest_float ('eta', 0.01, 0.5),
    #         'learning_rate':trial.suggest_float ('learning_rate', 0.001, 0.5),
    #         'max_depth':trial.suggest_int('max_depth', 3, 10),
    #         'reg_alpha':trial.suggest_float ('reg_alpha', 0.5, 10),
    #         'n_estimators':trial.suggest_int('n_estimators', 500, 2000),
    #         'reg_lambda':trial.suggest_float ('reg_lambda', 0.5, 10),
    #     }
    #     model = xgb.XGBRegressor(**params)
    #     model.fit(x_train, y_train)
    #     preds = model.predict(x_vaild)
    #     rmse = mean_absolute_error(y_vaild, preds)
    #     return rmse
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=10)
    # print("Number of finished trials: ", len(study.trials))
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # time_print("Tuning finished")

    
    # Training
    time_print("Training start")
    time_0 = time.time()
    # model = xgb.XGBRegressor(**study.best_trial.params)
    # model.fit(x_train, y_train)
    #Call the train function and specify the save_path if you want to save the learning curve image
    model = train(x_train, y_train, x_vaild, y_vaild, save_path='learning_curve.png')
    print("Training time: ", time.time() - time_0)
    time_print("Training finished")

    # print matrix
    time_print("Validation start")
    rmse_train = valid(model, x_train, y_train)
    rmse_vaild = valid(model, x_vaild, y_vaild)
    print("RMSE of training data: ", rmse_train)
    print("RMSE of validation data: ", rmse_vaild)
    time_print("Validation finished")

    # Get submission
    if args.get_submission:
        get_submission(model, x_test, args.submission_path)
        time_print("Submission file generated in {}".format(args.submission_path))