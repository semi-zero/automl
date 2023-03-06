import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging
from collections import defaultdict
import pickle

#전처리 class
class Preprocessing:
    def __init__(self, log_name, data, var_list, num_var, obj_var, target, unique_id, anomaly_per=10):
        self.df = data                                     # 데이터
        self.var_list = var_list                           # 전체 변수 리스트
        self.num_var = num_var                             # 수치형 변수 리스트
        self.obj_var = obj_var                             # 문자형 변수 리스트
        self.target = target                               # 타겟 변수
        self.unique_id = unique_id                         # id변수
        
        self._anomaly_ratio = int(anomaly_per)             # 지정 결측 범위
        self._anomaly_percentage = int(anomaly_per) / 100  # 지정 결측 범위
        #self.na_pre = na                                   #결측치 처리 여부
        
        self.logger = logging.getLogger(log_name)
        
        #결측치 처리 먼저 진행
        self.df = self.na_preprocess(self.df, self.num_var, self.obj_var, self._anomaly_ratio)
        
        self.df, self.id_target_df = self.id_target_preprocess(self.df, self.unique_id, self.target)
        
        # 표준화
        self.df = self.standardize(self.df, self.num_var)
        
        # 라벨 인코딩
        self.df = self.label_encoder(self.df, self.obj_var)
    
    
        # 결측치 확인 및 처리
    def na_preprocess(self, df, num_var, obj_var, anomaly_per):
        
        self.logger.info('결측치 처리')        
        
        try:
            obj_data = df.loc[:, obj_var]
            obj_data.fillna("NaN", inplace=True)

            num_data = df.loc[:, num_var] 
            num_data.fillna(num_data.mean(), inplace=True)

            df = pd.concat([obj_data, num_data], axis=1)
        except:
            self.logger.exception('결측치 처리에 문제가 발생하였습니다')
            
        self.logger.info(f'결측치 처리 이후 데이터 구성: {df.shape[0]} 행, {df.shape[1]}열')                  
        
        return df
    
  
    def id_target_preprocess(self, df, unique_id, target):
        
        self.logger.info('전처리를 위한 id와 target 분리')
        
        #식별 변수가 있을 수도 있고 없을 수도 있다(0119)
        
        try:
            id_target_df = df[[unique_id, target]]
            df = df.drop([unique_id, target], axis=1)
            
            if unique_id in self.num_var:
                self.num_var.remove(unique_id)    
            else : self.obj_var.remove(unique_id)
            
            if target in self.num_var:
                self.num_var.remove(target)
            else:
                lb_encoder = LabelEncoder()
                id_target_df.loc[:, target] = lb_encoder.fit_transform(id_target_df.loc[:, target])
                self.obj_var.remove(target)
            
        except:
            self.logger.exception('id 와 target 분리 처리에 문제가 발생하였습니다')
            
        return df, id_target_df
        
    
#     # 이상치 제거 절차 삭제(230119)
        
    
     #정규화
    def standardize(self, df, num_var):
                                  
        self.logger.info('정규화 진행')
        try:        
            if num_var:
                num_data = df.loc[:, num_var]
                non_num_data = df.drop(set(num_var), axis=1)

                #표준화
                std_scaler = StandardScaler()
                fitted = std_scaler.fit(num_data)
                output = std_scaler.transform(num_data)
                num_data = pd.DataFrame(output, columns = num_data.columns, index=list(num_data.index.values))

                tmp_df = pd.concat([non_num_data, num_data], axis=1)
            else:
                tmp_df = df
        except:
            self.logger.exception('정규화 진행 중에 문제가 발생하였습니다')                                      
                                  
        return tmp_df
        
    
    #문자형 변수를 수치형으로 변환
    def label_encoder(self, df, obj_var):
                                  
        self.logger.info('라벨 인코딩 진행')
        try:                              
            if obj_var:
                obj_data = df.loc[:, obj_var]
                non_obj_data = df.drop(set(obj_var), axis=1)

                #인코딩
                lbl_en = LabelEncoder()
                lbl_en = defaultdict(LabelEncoder)
                obj_data = obj_data.apply(lambda x:lbl_en[x.name].fit_transform(x))
            
                #라벨 인코딩 저장    
                pickle.dump(lbl_en, open('storage/model/label_encoder.sav', 'wb'))
                
            
                tmp_df = pd.concat([obj_data, non_obj_data], axis=1)
                
            else:
                tmp_df = df
                                  
        except:
            self.logger.exception('수치형 변환 중에 문제가 발생하였습니다')                                      
                                 
        return tmp_df
    
    
    def get_df(self):
        
        self.df = pd.concat([self.df, self.id_target_df], axis=1)
        
        self.logger.info('전처리 완료')
        self.logger.info('\n')
        self.logger.info(self.df.head())
        
        
        return self.df
    