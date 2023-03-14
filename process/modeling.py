import pandas as pd
import numpy as np
import datetime
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging

import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import torch.nn.functional as F
from . import hpo


class Modeling:
    def __init__(self, log_name, data, obj_var, target, unique_id, model_type='auto', OVER_SAMPLING=True, HPO=False):
        self.ori_df = data                          #데이터
        self.target = target                    #타겟 변수 지정
        self.unique_id = unique_id
        self.over_sampling = OVER_SAMPLING      #오버 샘플링 여부
        self.hpo = HPO                          #HPO 여부
        self.obj_var = obj_var
        self.model_type = model_type
        self.model = dict()
        self.score = dict()
        self.test = dict()
        
        self.logger = logging.getLogger(log_name)
        
        self.start_time = datetime.datetime.now()
        
        #모델링 준비
        self.df = self.id_modeling(self.ori_df, self.unique_id)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(self.df, self.target, self.over_sampling)
        
        # 결과값 딕셔너리
        self.score['AUROC']     = dict()
        self.score['AUCPR']     = dict()
        self.score['정확도']  = dict()
        self.score['정밀도'] = dict()
        self.score['재현율']    = dict()
        self.score['F1']        = dict()
        
        
        #모델링 딕셔너리
        model_type_dict = {'lr'  : self.lr_process(),
                           'rf'  : self.rf_process(),
                           'lgb' : self.lgb_process(),
                           'tab' : self.tab_process()}
        
        self.best_model_name, self.best_model, self.best_test = self.get_best_model()
       
        #학습 결과 화면을 위한 함수들
        #1. 분석 리포트
        self.report = self.make_report(self.target, self.model_type, self.over_sampling, self.hpo, self.start_time)
        
        #2. 학습 결과 비교 화면
        #3. 변수 중요도
        self.test_score, self.valid_score, self.fi = self.get_eval(self.best_model_name, self.best_model, self.best_test, self.ori_df, self.unique_id, self.target)
        
        
        self.to_result_page()
        

            
    def id_modeling(self, df, unique_id):
        
        self.logger.info('모델링을 위한 id 분리')
        try:
            id_df = df[[unique_id]]
            df = df.drop([unique_id], axis=1)
            
        except:
            self.logger.exception('모델링을 위한 id분리에 문제가 발생하였습니다')
        
        return df
    
    
    #train, valid 분리
    def train_test_split(self, df, target, over_sampling):
        
        self.logger.info('train valid 분리')
        
        try:
            df_y = pd.DataFrame(df.loc[:,target])
            df_x = df.drop(target, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=20)
            
            
            try:
                if over_sampling:
                    self.logger.info('불균형 처리')
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
            except:
                self.logger.exception('불균형 처리에 문제가 발생하였습니다.')
                
        except:
            self.logger.exception('train valid 분리에 문제가 발생하였습니다')
        
        self.logger.info('train valid 분리 후')
        self.logger.info(f'X_train.shape: {X_train.shape}')
        self.logger.info(f'X_test.shape : {X_test.shape}')
        self.logger.info(f'y_train.shape: {y_train.shape}')
        self.logger.info(f'y_test.shape : {y_test.shape}')
        
        return X_train, X_test, y_train, y_test
    
     #################### lR START####################
    
    def lr_process(self):
        if (self.model_type == 'lr') or (self.model_type == 'auto'): 
            self.lr_fit_predict(self.X_train, self.y_train, self.X_test, self.y_test, self.hpo)
        
        
    #모델 fit
    def lr_fit_predict(self, X_train, y_train, X_test, y_test, HPO):
        
        
        self.logger.info('lr 모델 hpo 세팅')
        try:
                        
            if HPO :
                self.logger.info('lr HPO 진행') 
                parameters = hpo.HyperOptimization(X = X_train, y = y_train, model = 'lr').best_params
                self.logger.info(f'lr HPO 진행 후 parameters: {parameters}')
                
            else: 
                parameters ={'C':1.0,  'class_weight':None, 'solver':'lbfgs'}
                
            self.logger.info(f'세팅된 parameters : {parameters}')
                
        except:
            self.logger.exception('lr 모델 hpo 세팅에 실패했습니다') 
        
        #scoring 
        self.logger.info('lr 모델 fitting')
        try:
            
            lr = LogisticRegression(**parameters)
            lr.fit(X_train, y_train)
            y_pred_proba = lr.predict_proba(X_test)[:,1]
            y_pred = lr.predict(X_test)
            
            self.model['lr'] = lr
            self.test['lr'] = (X_test, y_test)

            self.score['AUROC']['lr']     = np.round(roc_auc_score(y_test, y_pred_proba), 3)
            self.score['AUCPR']['lr']     = np.round(average_precision_score(y_test, y_pred_proba), 3)
            self.score['정확도']['lr']    = np.round(accuracy_score(y_test, y_pred), 3)
            self.score['정밀도']['lr']    = np.round(precision_score(y_test, y_pred), 3)
            self.score['재현율']['lr']    = np.round(recall_score(y_test, y_pred), 3)
            self.score['F1']['lr']        = np.round(f1_score(y_test, y_pred), 3)
            
        except:
            self.logger.exception('lr 모델 fitting에 실패했습니다')     
    
   #################### LR FINISH ####################    
    
    
    #################### RF START####################
    
    def rf_process(self):
        if (self.model_type == 'rf') or (self.model_type == 'auto'): 
            self.rf_fit_predict(self.X_train, self.y_train, self.X_test, self.y_test, self.hpo)
        
        
    #모델 fit
    def rf_fit_predict(self, X_train, y_train, X_test, y_test, HPO):
        
        
        self.logger.info('rf 모델 hpo 세팅')
        try:
                        
            if HPO :
                self.logger.info('rf HPO 진행') 
                parameters = hpo.HyperOptimization(X = X_train, y = y_train, model = 'rf').best_params
                self.logger.info(f'rf HPO 진행 후 parameters: {parameters}')
                
            else: 
                parameters ={'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
                
            self.logger.info(f'세팅된 parameters : {parameters}')
                
        except:
            self.logger.exception('rf 모델 hpo 세팅에 실패했습니다') 
        
        #scoring 
        self.logger.info('rf 모델 fitting')
        try:
            
            rf = RandomForestClassifier(**parameters)
            rf.fit(X_train, y_train)
            y_pred_proba = rf.predict_proba(X_test)[:,1]
            y_pred = rf.predict(X_test)
            
            self.model['rf'] = rf
            self.test['rf'] = (X_test, y_test)

            self.score['AUROC']['rf']     = np.round(roc_auc_score(y_test, y_pred_proba), 3)
            self.score['AUCPR']['rf']     = np.round(average_precision_score(y_test, y_pred_proba), 3)
            self.score['정확도']['rf']    = np.round(accuracy_score(y_test, y_pred), 3)
            self.score['정밀도']['rf']    = np.round(precision_score(y_test, y_pred), 3)
            self.score['재현율']['rf']    = np.round(recall_score(y_test, y_pred), 3)
            self.score['F1']['rf']        = np.round(f1_score(y_test, y_pred), 3)
            
        except:
            self.logger.exception('rf 모델 fitting에 실패했습니다')     
    
   #################### RF FINISH ####################    
    
   
    #################### lGBM START####################
    
    def lgb_process(self):
        if (self.model_type == 'lgb') or (self.model_type == 'auto'): 
            self.lgb_fit_predict(self.X_train, self.y_train, self.X_test, self.y_test, self.hpo)
        
        
    #모델 fit
    def lgb_fit_predict(self, X_train, y_train, X_test, y_test, HPO):
        
        
        self.logger.info('lgbm 모델 hpo 세팅')
        try:
                        
            if HPO :
                self.logger.info('lgb HPO 진행') 
                parameters = hpo.HyperOptimization(X = X_train, y = y_train, model = 'lgb').best_params
                self.logger.info(f'lgb HPO 진행 후 parameters: {parameters}')
                
            else: 
                parameters ={'num_leaves':[60], 'min_child_samples':[10],'max_depth':[5],'learning_rate':[0.1],'reg_alpha':[0.01]}
                
            self.logger.info(f'세팅된 parameters : {parameters}')
                
        except:
            self.logger.exception('lgbm 모델 hpo 세팅에 실패했습니다') 
        
        #scoring 
        self.logger.info('lgbm 모델 fitting')
        try:
            
            lgb = LGBMClassifier(**parameters)
            lgb.fit(X_train, np.ravel(y_train, order='C'))
            y_pred_proba = lgb.predict_proba(X_test)[:,1]
            y_pred = lgb.predict(X_test)
            
            self.model['lgb'] = lgb
            self.test['lgb'] = (X_test, y_test)

            self.score['AUROC']['lgb']     = np.round(roc_auc_score(y_test, y_pred_proba), 3)
            self.score['AUCPR']['lgb']     = np.round(average_precision_score(y_test, y_pred_proba), 3)
            self.score['정확도']['lgb']    = np.round(accuracy_score(y_test, y_pred), 3)
            self.score['정밀도']['lgb']    = np.round(precision_score(y_test, y_pred), 3)
            self.score['재현율']['lgb']    = np.round(recall_score(y_test, y_pred), 3)
            self.score['F1']['lgb']        = np.round(f1_score(y_test, y_pred), 3)
            
        except:
            self.logger.exception('lgbm 모델 fitting에 실패했습니다')     
    
   #################### lGBM FINISH ####################



    #################### TabNet START ####################
    def tab_process(self):
        if (self.model_type == 'tab') or (self.model_type == 'auto'): 
            self.cat_idxs, self.cat_dims = self.get_tabnet_param()
            self.tab_fit_predict(self.cat_idxs, self.cat_dims, self.X_train, self.y_train, self.X_test, self.y_test, self.hpo)
    
    
    def get_tabnet_param(self):
        
        self.logger.info('tabnet params 세팅')
        try:                                  
            nuinque = self.df.nunique()
            types = self.df.dtypes

            categorical_columns = []
            categorical_dims = {}
            target = self.target

            for col in self.obj_var:
                categorical_columns.append(col)
                categorical_dims[col] = self.df[col].nunique()

            features = [col for col in self.df.columns if col not in [target]]
            cat_idxs = [i for i, f in enumerate(features) if f in self.obj_var]
            cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

            tabnet_param = {'cat_idxs' : cat_idxs, 'cat_dims':cat_dims}
            self.logger.info(f'tabnet_param : {tabnet_param}')
            
            self.logger.info('tabnet params를 object storage에 저장')                                  
            try:                                   
                with open('storage/model/tabnet_param.json', 'w') as f:
                    json.dump(tabnet_param, f)
            except:
                self.logger.exception('tabnet params를 object storage에 저장하기가 실패했습니다')                                 
                                                                    
        except:
            self.logger.exception('tabnet params 세팅에 실패했습니다')           

        return cat_idxs, cat_dims

    
    def tab_fit_predict(self, cat_idxs, cat_dims, X_train, y_train, X_test, y_test, HPO):
        
        self.logger.info('tabnet 모델 세팅')
        try:
            X_train_values = X_train.values
            y_train_values = y_train.values.reshape(-1,)
            X_test_values = X_test.values
            y_test_values = y_test.values.reshape(-1,)
            
            
            if HPO :
                self.logger.info('tab HPO 진행') 
                parameters = hpo.HyperOptimization(X = X_train, y = y_train, model = 'tab').best_params
                self.logger.info(f'tab HPO 진행 후 parameters: {parameters}')                   
            
            else: 
                parameters ={'mask_type': 'entmax', 'n_da': 8, 'n_steps': 3, 'gamma': 1.3, 'n_shared': 2, 
                              'lambda_sparse': 1e-3, 'patience': 10, 'epochs': 2}
                
            self.logger.info(f'세팅된 parameters : {parameters}')
                                  
        except:
            self.logger.exception('tabnet 모델 세팅에 실패했습니다')                                     

            
        self.logger.info('tabnet 모델 fitting')
        try:                                  

            tab=TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=10, 
                     optimizer_fn = torch.optim.Adam, optimizer_params=dict(lr=1e-2),
                     n_d = parameters['n_da'], n_a = parameters['n_da'], n_steps = parameters['n_steps'], n_shared = parameters['n_shared'], 
                     lambda_sparse = parameters['lambda_sparse'], gamma = parameters['gamma'],
                     scheduler_params={'step_size':50},
                     scheduler_fn = torch.optim.lr_scheduler.StepLR, mask_type = parameters['mask_type']) 

            tab.fit(X_train=X_train_values, y_train=y_train_values,
                   eval_set=[(X_train_values, y_train_values), (X_test_values, y_test_values)], eval_name=['train', 'valid'], eval_metric=['auc'],
                    max_epochs=parameters['epochs'], patience=parameters['patience'], 
                    batch_size=1024, virtual_batch_size =128, num_workers=1, weights=1, drop_last=False,)# loss_fn=focal_loss)

            y_pred_proba = tab.predict_proba(X_test_values)[:,1]
            y_pred = tab.predict(X_test_values)
            
            self.model['tab'] = tab
            self.test['tab'] = (X_test_values, y_test_values)   
            
            self.score['AUROC']['tab']     = np.round(roc_auc_score(y_test, y_pred_proba), 3)
            self.score['AUCPR']['tab']     = np.round(average_precision_score(y_test, y_pred_proba), 3)
            self.score['정확도']['tab']    = np.round(accuracy_score(y_test, y_pred), 3)
            self.score['정밀도']['tab']    = np.round(precision_score(y_test, y_pred), 3)
            self.score['재현율']['tab']    = np.round(recall_score(y_test, y_pred), 3)
            self.score['F1']['tab']        = np.round(f1_score(y_test, y_pred), 3)
            
            
        
        except:
            self.logger.exception('tabnet 모델 fitting에 실패했습니다')                
        
    #################### TabNet FINISH####################
    
    def get_best_model(self):
        
        self.logger.info('Auto ML 가동')
        self.logger.info(f'automl_score:{self.score}')
        try:
            best_model_name = max(self.score['AUROC'], key=self.score['AUROC'].get) 
            best_model = self.model[best_model_name]
            best_test = self.test[best_model_name]
            
            self.logger.info(f'best_model_name: {best_model_name}')
                                  
        except:
            self.logger.exception('best 모델 선정에 실패했습니다')                                                                 
        
        
        self.logger.info('best 모델 저장')
        try:
            # 모델 저장
            joblib.dump(best_model, 'storage/model/best_model.pkl')

        except:
            self.logger.exception('best 모델 저장에 실패했습니다')                                                                 
        
        return best_model_name, best_model, best_test
        
        
    def make_report(self, target, model_type, over_sampling, hpo, start_time):
        
        self.logger.info('학습 결과를 위한 결과물 생성')
        try:
            report = pd.DataFrame({'상태' : '완료됨',
                                  '모델 ID' : ['model_id'],
                                  '생성 시각': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
                                  '학습 시간' : [datetime.datetime.now()-start_time],
                                   '데이터셋 ID' : 'dataset_id',
                                   '타겟 변수' : target,
                                   '데이터 분할' : '80/20',
                                   '알고리즘' : model_type, 
                                   '목표' : '테이블 형식 분류',
                                   '최적화 목표' : 'AUROC',
                                   '불균형 처리 여부' : over_sampling,
                                   'HPO 여부' : hpo})
            report = report.T
        
        except:
            self.logger.exception('학습 결과를 위한 결과물 생성 실패했습니다')
            
        return report
            
            
    def get_eval(self, best_model_name, best_model, best_test, ori_df, unique_id, target):
        
        self.logger.info('best 모델 검증')
        
        try:
            X_test, y_test = best_test[0], best_test[1]
            pred_proba = best_model.predict_proba(X_test)[:,1]
            pred = best_model.predict(X_test)
        
            self.logger.info(f'오차행렬 \n : {confusion_matrix(y_test, pred)}')
            self.logger.info(f'정확도 : {accuracy_score(y_test, pred)}')
            self.logger.info(f'정밀도 : {precision_score(y_test, pred)}')
            self.logger.info(f'재현율 : {recall_score(y_test, pred)}')
            self.logger.info(f'f1 score : {f1_score(y_test, pred)}')
            self.logger.info(f'roc auc score : {roc_auc_score(y_test, pred_proba)}')
            
            test_score = pd.DataFrame({'오차행렬' : [confusion_matrix(y_test, pred)],
                                       '정확도' : [np.round(accuracy_score(y_test, pred),3)],
                                       '정밀도' : [np.round(precision_score(y_test, pred),3)],
                                       '재현율' : [np.round(recall_score(y_test, pred),3)],
                                       'F1' : [np.round(f1_score(y_test, pred),3)],
                                       'AUROC' : [np.round(roc_auc_score(y_test, pred_proba),3)]
                                      })
        
                                       
        except:
            self.logger.exception('best 모델 검증에 실패했습니다')
       
    
        self.logger.info('모델별 학습 결과표 생성')    
        
        try:
            valid_score = pd.DataFrame(self.score)
            
        except:
            self.logger.exception('모델별 학습 결과표를 저장했습니다')
            
                                  
                                  
        self.logger.info('변수 중요도 저장')
        try:
            
            id_df = ori_df[[unique_id]]
            df = ori_df.drop([unique_id, target], axis=1)
            print(df.shape)
            
            #변수 중요도 저장
            if 'tab' in best_model_name: 
                feat_importances = best_model.feature_importances_
                
                
                #tabnet의 경우 instance별 변수 중요도를 도출할 수 있음
                explain, _ = best_model.explain(df.values)
                instance_ = pd.DataFrame(explain, columns = df.columns)
                instance_importance = instance_.div(instance_.sum(axis=1), axis=0)
                instance_importance = pd.concat([id_df, instance_importance], axis=1)
                instance_importance.to_csv('instance_importance.csv', index=False)
                
            else : 
                feat_importances = best_model.feature_importances_
            
            
            #fi_df = pd.DataFrame(fi.reshape(1,-1), columns = X_test.columns)
            fi_df = pd.DataFrame(feat_importances, index=df.columns).sort_values(0, ascending=False)

        except:
            self.logger.exception('모델 결과값들을 저장했습니다')
            
    
        return test_score, valid_score, fi_df
    
    
    def to_result_page(self):
        
        train_result_page = {'best_model_name': self.best_model_name,
                            'score': self.score}
        
        save_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open('storage/'+str(save_time)+'result_page_path', 'w') as f:
            json.dump(train_result_page, f)
        