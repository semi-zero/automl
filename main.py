import argparse
import logging
import pandas as pd
import random
import os
import numpy as np
from loggers import logger
from process import input_data, modeling, preprocess, check_reduce

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')

# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')
parser.add_argument('-target', '--target', type=str, help='Target vairable')
parser.add_argument('-id', '--unique_id', type=str, help='ID variavle')
parser.add_argument('-over', '--OVER_SAMPLING', action='store_true', help='oversample process bool')
parser.add_argument('-model_type', '--model_type', type=str, help='model type')
parser.add_argument('-hpo', '--HPO', action='store_true', help='hyperparameter optimize bool')


args = parser.parse_args()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore

def set_logger(log_name):
    log_obj = logger.AutoMLLog(log_name)
    log_obj.set_handler('automl_process')
    log_obj.set_formats()
    auto_logger = log_obj.addOn()
    
    auto_logger.info('logger 세팅')
        

if __name__ == "__main__":
    seed_everything()
    log_name = 'automl_tabular'
    set_logger(log_name)
    data, var_list, num_var, obj_var = input_data.Data_load(args.PATH, log_name).read_data()
    check = check_reduce.Data_check_reduce(log_name, data, args.target, args.unique_id).check
    if check == True:
        df = preprocess.Preprocessing(log_name, data, var_list, num_var, obj_var, args.target, args.unique_id).get_df()
        mm = modeling.Modeling(log_name, df, obj_var = obj_var, target=args.target, unique_id = args.unique_id, model_type=args.model_type, OVER_SAMPLING=args.OVER_SAMPLING, HPO=args.HPO)
    
    
    # 입력 예시
    # python main.py -pth storage/data/churn.csv -target churn -id state -over -model_type auto -hpo
    # python main.py -pth storage/data/churn2.csv -target Churn -id customerID -over -model_type auto -hpo 
    # python main.py -pth storage/data/credit.csv -target Class -id Time -over -model_type auto -hpo 
    # python main.py -pth storage/data/insurance.csv -target fraud_reported -id policy_number -over -model_type auto -hpo 
    # python main.py -pth storage/data/synthetic.csv -target isFraud -id step -over -model_type auto -hpo 
    