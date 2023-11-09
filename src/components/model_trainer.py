import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    pass

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor() 
            }

            params = {
                "Linear Regression": {},
                "Ridge Regression": {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
                },
                "Lasso Regression": {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'n_estimators': list(range(500,1000,100)),
                    'max_depth': list(range(4,9,4)),
                    'min_samples_split': list(range(4,9,2)),
                    'min_samples_leaf': [1,2,5,7],
                    'max_features':['auto','sqrt']
                    
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            
        except Exception as e:
            raise CustomException(e,sys)
        




