import pandas as pd
import logging
import numpy as np

class Data_load:
    def __init__(self, path, log_name):
        self.path = path #데이터 위치 경로 입력
        self.logger = logging.getLogger(log_name)

    def read_data(self):

        self.logger.info('csv 데이터 불러오기')
        self.logger.info(f'{self.path}')
        
        #코드 수정해야 함 
        try:
            df = pd.read_csv(self.path)
        except:
            try:
                df = pd.read_csv(self.path, encoding='cp949')
            except:
                try:
                    df = pd.read_csv(self.path, encoding='utf-8')
                except:
                    self.logger.info('데이터 포맷을 맞춰주세요')
        
        self.logger.info('변수 분리 시작')
        
        try:
            df = df.sort_index(axis=1)
            var_list = df.columns.tolist() #전체 변수리스트 추출
            num_var = df.select_dtypes(include='float').columns.tolist() + df.select_dtypes(include='int').columns.tolist() #수치형 변수 추출
            obj_var = [x for x in df.columns if x not in num_var] #문자형 변수 추출
        
        except: 
            self.logger.error('csv 데이터 불러오기를 실패했습니다')
        
        
        
        return df, var_list, num_var, obj_var
    
    
