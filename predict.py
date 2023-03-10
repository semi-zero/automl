import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse

import joblib
import json
import glob
import logging
from loggers import logger
import pickle

from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')

# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')
parser.add_argument('-storage', '--storage',  type=str, help='Path of Storage')
parser.add_argument('-id', '--unique_id', type=str, help='ID variavle')

args = parser.parse_args()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def set_logger(log_name):
    log_obj = logger.AutoMLLog(log_name)
    log_obj.set_handler('predict_process')
    log_obj.set_formats()
    auto_logger = log_obj.addOn()
    
    auto_logger.info('logger 세팅')
    
    
class Predict:
    def __init__(self, test_path, storage_path, unique_id):
        self.test_path = test_path
        self.storage_path = storage_path
        self.unique_id = unique_id
        self.logger = logging.getLogger(log_name)
        
        self.test_ori_df, self.var_list, self.num_var, self.obj_var = self.read_test_data(self.test_path)
        
        self.test_df = self.id_predicting(self.test_ori_df, self.unique_id) 
                                         
        # 표준화
        self.test_df = self.standardize(self.test_df, self.num_var)
        
        # 라벨 인코딩
        self.test_df = self.label_encoder(self.test_df, self.obj_var, self.storage_path)
        
        self.result = self.get_pred_df(self.test_df, self.storage_path, self.test_ori_df)
        
        
    # 데이터 불러오기
    def read_test_data(self, test_path):
        
        #코드 수정해야 함 
        
        
        try:
            test_df = pd.read_csv(test_path)
            
        except:
            
            try:
                test_df = pd.read_csv(test_path, encoding='cp949')
            
            except:
                
                try:
                    test_df = pd.read_csv(test_path, encoding='utf-8')
                
                except:
                    self.logger.info('데이터 포맷을 맞춰주세요')
            
        try:
            var_list = test_df.columns.tolist() #전체 변수리스트 추출
            num_var = test_df.select_dtypes(include='float').columns.tolist() + test_df.select_dtypes(include='int').columns.tolist() #수치형 변수 추출
            obj_var = [x for x in test_df.columns if x not in num_var] #문자형 변수 추출
            
        except:
            self.logger.exception('test 데이터 불러오기가 실패했습니다')
            
        return test_df, var_list, num_var, obj_var
    
    
    
    #테스트 데이터 정합성
    
    
    def id_predicting(self, df, unique_id):
        
        self.logger.info('예측을 위한 id 분리')
        try:
            id_df = df[[unique_id]]
            df = df.drop([unique_id], axis=1)
            if unique_id in self.num_var:
                self.num_var.remove(unique_id)    
            else : self.obj_var.remove(unique_id)    
        
        except:
            self.logger.exception('예측을 위한 id분리에 문제가 발생하였습니다')
        
        return df

    def standardize(self, test_df, num_var):
        
        self.logger.info('예측데이터 표준화')
        try:
            num_data = test_df.loc[:, num_var]
            non_num_data = test_df.drop(set(num_var), axis=1)
        
            #표준화
            std_scaler = StandardScaler()
            fitted = std_scaler.fit(num_data)
            output = std_scaler.transform(num_data)
            num_data = pd.DataFrame(output, columns = num_data.columns, index=list(num_data.index.values))

            tmp_df = pd.concat([non_num_data, num_data], axis=1)
            
        except:
            self.logger.exception('예측 데이터 표준화에 문제가 발생하였습니다')
        
        return tmp_df
        
    
    #문자형 변수를 수치형으로 변환
    def label_encoder(self, test_df, obj_var, storage_path):
        
        self.logger.info('예측데이터 인코딩')
        
        try:
            obj_data = test_df.loc[:, obj_var]
            non_obj_data = test_df.drop(set(obj_var), axis=1)
            
            #라벨 인코더 불러오기
            self.logger.info('라벨 인코더 불러오기')
            
            with open(f'{storage_path}/label_encoder.pkl', mode='rb') as f:
                test_enc = pickle.load(f)
            
            for var in obj_var:
                test_enc[var].classes_ = np.append(test_enc[var].classes_, '<unkown>')
            
            obj_data = obj_data.apply(lambda x:test_enc[x.name].transform(x))
            tmp_df = pd.concat([obj_data, non_obj_data], axis=1)
        
        except:
            self.logger.exception('예측데이터 인코딩에 문제가 발생했습니다.')
            
        return tmp_df
    
    
    def get_pred_df(self, test_df, storage_path, test_ori_df):
        
        self.logger.info('모델 불러오기 시작')
        
        try:
            
            model_path = glob.glob(storage_path+'/*model*')[0]
            load_model=joblib.load(model_path)
        
#             name_path = glob.glob(storage_path+'/*name*')[0]
            
#             with open(name_path) as f:
#                 model_name = json.load(f)
            
#             self.logger.info(f"model_name : {model_name['best_model_name']}")
            
            try:
                self.logger.info('예측 시작')
                pred = load_model.predict(test_df.values)
                pred_proba = load_model.predict_proba(test_df.values)
                
            except:
                
                self.logger.exception('예측 실패') 
                
            test_ori_df['pred'] = pred
            test_ori_df['pred_proba'] = pred_proba[:,1]
        
            #prediction 저장
            test_ori_df.to_csv('prediction.csv', index=False)
        
        except:
            self.logger.exception('모델 불러오기 실패')
        
        return test_ori_df
    

if __name__ == "__main__":
    log_name = 'automl_tabular_predict'
    set_logger(log_name)
    Predict(test_path=args.PATH, storage_path=args.storage, unique_id=args.unique_id)
    
    #python predict.py -pth storage/data/churn_test.csv -storage storage/model -id state
    #python predict.py -pth storage/data/churn2_test.csv -storage storage/model -id customerID
    
    