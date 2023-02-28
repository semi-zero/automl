import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import torch.nn.functional as F

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class HyperOptimization:
    
    def __init__(self, X, y, model):
     
        self.X_train = X
        self.y_train = y
        self.model = model

        self.obj = {'lr'  : self.lr_objective,
                    'rf'  : self.rf_objective,
                    'lgb' : self.lgb_objective,
                    'tab' : self.tab_objective}

        sampler = TPESampler(seed=42)
        study = optuna.create_study(study_name="parameter_opt", direction="maximize", sampler=sampler,)
        study.optimize(self.obj[self.model], n_trials=10)
        self.best_params = study.best_params

    def lr_objective(self, trial):
        
        params_lr = {
        "C": trial.suggest_float('C', 8e-3, 0.1),
        "max_iter": 1000,
        #"class_weight": {0:1, 1:trial.suggest_float('class_weight', 1,1.5)},
        "solver": trial.suggest_categorical('solver', ['liblinear']),
        "random_state": 42,
        }
        
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        CV_score_array = []
        
        for n_fold, (train_index, val_index) in enumerate(folds.split(self.X_train, self.y_train)):
            train_x, val_x = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
            train_y, val_y = self.y_train.iloc[train_index, :], self.y_train.iloc[val_index, :] 
            lr_clf = LogisticRegression(**params_lr)
            lr_clf.fit(train_x, train_y) #early_stopping_rounds=100, verbose=False,

            
            lr_pred = lr_clf.predict_proba(val_x)[:,1]
            lr_score = roc_auc_score(val_y, lr_pred)
    
            CV_score_array.append(lr_score)
        avg = np.mean(CV_score_array)
        
        print(f'avg : {avg}')
        
        return avg
    
    def rf_objective(self, trial):
        
        params_rf = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }
        
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        CV_score_array = []
        
        for n_fold, (train_index, val_index) in enumerate(folds.split(self.X_train, self.y_train)):
            train_x, val_x = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
            train_y, val_y = self.y_train.iloc[train_index, :], self.y_train.iloc[val_index, :] 
            rf_clf = RandomForestClassifier(**params_rf)
            rf_clf.fit(train_x, train_y) #early_stopping_rounds=100, verbose=False,

            
            rf_pred = rf_clf.predict_proba(val_x)[:,1]
            rf_score = roc_auc_score(val_y, rf_pred)
    
            CV_score_array.append(rf_score)
        avg = np.mean(CV_score_array)
        
        print(f'avg : {avg}')
        
        return avg
            
    def lgb_objective(self, trial):
        
        params_lgb = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024, step=1, log=True), 
            'max_depth': trial.suggest_int('max_depth', 1, 10, step=1, log=False), 
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True), 
            'n_estimators': trial.suggest_int('n_estimators', 8, 1024, step=1, log=True), 
            'objective': 'binary', "metric": "auc",
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, step=1, log=False), 
            'subsample': trial.suggest_uniform('subsample', 0.7, 1.0), 
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),
            'random_state': 0
        }

        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        CV_score_array = []
        
        for n_fold, (train_index, val_index) in enumerate(folds.split(self.X_train, self.y_train)):
            train_x, val_x = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
            train_y, val_y = self.y_train.iloc[train_index, :], self.y_train.iloc[val_index, :] 
            lgb_clf = LGBMClassifier(**params_lgb)
            lgb_clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)], ) #early_stopping_rounds=100, verbose=False,

            
            lgb_pred = lgb_clf.predict_proba(val_x)[:,1]
            lgb_score = roc_auc_score(val_y, lgb_pred)
    
            CV_score_array.append(lgb_score)
        avg = np.mean(CV_score_array)
        
        print(f'avg : {avg}')
        
        return avg
    
    
    def tab_objective(self, trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patienceScheduler",low=3,high=10), 
                                               min_lr=1e-5,
                                               factor=0.5,),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         ) #early stopping

        folds=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        CV_score_array  =[]

        self.X_train_values = self.X_train.values
        self.y_train_values = self.y_train.values.reshape(-1,)
        
        for n_fold, (train_index, val_index) in enumerate(folds.split(self.X_train_values, self.y_train_values)):

            train_x, val_x = self.X_train_values[train_index], self.X_train_values[val_index]
            train_y, val_y = self.y_train_values[train_index], self.y_train_values[val_index]

            tab_clf = TabNetClassifier(**tabnet_params)
            tab_clf.fit(X_train=train_x, y_train=train_y,
                      eval_set=[(val_x, val_y)],
                      patience=trial.suggest_int("patience",low=15,high=30), max_epochs=trial.suggest_int('epochs', 1, 100),
                      eval_metric=['auc'])
            CV_score_array.append(tab_clf.best_cost)
        avg = np.mean(CV_score_array)
        return avg