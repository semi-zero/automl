import pandas as pd
import logging
import numpy as np

class Data_check_reduce:
    def __init__(self, log_name, data, target, unique_id):
        self.logger = logging.getLogger(log_name)
        
        self.df = data
        self.target = target
        self.unique_id = unique_id
        
        self.logger = logging.getLogger(log_name)
        
        #데이터 check가 통과되지 못하면 False로 변경
        self.check = True
        
        #데이터 check
        self.data_check()
        
        if self.check == True:
            self.reduce_mem_usage(self.df)
        
    def data_check(self):
        
        self.logger.info('데이터 정합성 확인 절차 시작')        
        
        
        if self.df[self.target].isnull().sum() != 0:
            self.logger.info('타겟 변수에 결측치가 포함되어 있습니다')
            self.check = False
        
        if self.df[self.unique_id].isnull().sum() != 0:
            self.logger.info('타겟 변수에 결측치가 포함되어 있습니다')
            self.check = False
            
        if self.df[self.target].nunique() != 2:
            self.logger.info('타겟 변수가 이진 변수가 아닙니다')
            self.check = False
        
        self.logger.info('데이터 정합성 확인 절차 종료')
        self.logger.info(f'데이터 정합성 확인 절차 결과 : {self.check}')    
    
    
    #데이터 메모리 줄이기
    def reduce_mem_usage(self, df):
        """ 
        iterate through all the columns of a dataframe and 
        modify the data type to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'데이터 구성: {df.shape[0]} 행, {df.shape[1]}열')
        self.logger.info(f'Memory usage of dataframe is {start_mem:.2f}MB')
    
        for col in df.columns:
            col_type = df[col].dtype
        
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max <\
                    np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max <\
                    np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max <\
                    np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max <\
                    np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max <\
                    np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max <\
                    np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    pass
            else:
                df[col] = df[col].astype('category')
        end_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'Memory usage after optimization is: {end_mem:.2f}MB')
        self.logger.info(f'Decreased by {100*((start_mem - end_mem)/start_mem):.1f}%')
    
        return df