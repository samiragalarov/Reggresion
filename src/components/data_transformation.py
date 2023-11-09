import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from src.exception import CustomException
from src.logger import logging
import os
from sklearn.preprocessing import FunctionTransformer
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder



# **Drop Car name**
# Finding outliers with IR
# Replaceing Year with Age
# Label Encoding
# One hot encoding 

out_arr = []
class Replace_Year_Age(BaseEstimator,TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    def transform(self , X):
        X['Age'] = 2020 - X['Year']
        X =  X.drop('Year',axis=1,inplace = False)
        return X


class Drop_Car_name(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop('Car_Name', axis=1)
        return X


    
class Clean_outlier(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self , X):


        columns_outlier = ['Kms_Driven' ,'Present_Price','Age']

        for i in columns_outlier:
            sorted(X[i])
            quantile1 , quantile3 = np.percentile(X[i], [25,75])
            iqr_value = quantile3 - quantile1

            lower_bound_val = quantile1 - (1.5 * iqr_value)
            upper_bound_val = quantile3 + (1.5 * iqr_value)

            a = X[(X[i] > upper_bound_val) & (X[i] > lower_bound_val)].index
            out_arr.append(list(a))

            X = X[(X[i] < upper_bound_val ) & (X[i]> lower_bound_val)]
            
            return X


class Label_encoder(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self , X):
        label_encoder = LabelEncoder()
        encoded_arr = label_encoder.fit_transform(X['Transmission'])
        X['Transmission'] = encoded_arr
        return X         

class Ohe_encoder(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self , X):
        X = pd.get_dummies( X,prefix =['Fuel_Type','Seller_Type'],dtype=int )
        # columns_arr = X.columns.values
        print(X)
        return X  




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        

    def get_data_transformer_object(self):

        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["Year","Selling_Price", "Present_Price"]
            categorical_columns = [
                "Fuel_Type",
                "Seller_Type",
                "Transmission"
            ]

            transform_pipeline = Pipeline(
                steps=[
                    ("Replace_Year_Age",Replace_Year_Age()),
                    ("Drop_Car_name",Drop_Car_name()),
                    ("Clean_outlier",Clean_outlier()),
                    ("Label_encoder",Label_encoder()),
                    ("Ohe_encoder",Ohe_encoder())                     
                ]
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("transform_pipeline",transform_pipeline,make_column_selector())
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
       


            target_column_name="Selling_Price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            preprocessing_obj = preprocessing_obj.fit(input_feature_train_df)

            target_feature_train_df.drop(out_arr[0], inplace = True)
            target_feature_test_df.drop(out_arr[1], inplace = True)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
    

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
      