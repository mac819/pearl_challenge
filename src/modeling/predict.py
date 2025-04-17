import pickle
import pandas as pd
import xgboost as xgb
from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def load_data(data_path, processed_util_path):
    test_data = pd.read_csv(data_path)
    index_test_data = test_data[['FarmerID']]

    with open(processed_util_path, 'rb') as f:
        processing_util = pickle.load(f)

    categorical_cols = processing_util['categorical_columns']

    test_dmatrix = test_data.drop(['FarmerID'], axis=1)
    test_dmatrix = xgb.DMatrix(test_dmatrix)
    return test_dmatrix, index_test_data

def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


if __name__ == "__main__":
    exp_name = 'exp_v1'
    test_data_path = PROCESSED_DATA_DIR / exp_name / 'test_v0.csv'
    processed_util_path = PROCESSED_DATA_DIR / exp_name / 'processing_util.pkl'
    model_path = MODELS_DIR / exp_name / 'income_regression_model.bst'

    print('Load Dataset')
    test_dmatrix, index_test_data = load_data(data_path=test_data_path, processed_util_path=processed_util_path)
    
    print('Load model')
    model = load_model(model_path=model_path)
    y_pred = model.predict(test_dmatrix)
    index_test_data['farmer_pred_income'] = y_pred

    index_test_data.to_csv('test_submission.csv', index=False)
