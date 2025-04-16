import optuna
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROCESSED_DATA_DIR

# Load sample data
def load_data(data_dir):
    train_data_path = data_dir / 'train_v0.csv'
    processing_util_path = data_dir / 'processing_util.pkl'

    with open(processing_util_path, 'rb') as f:
        processing_util = pickle.load(f)
    data = pd.read_csv(train_data_path)

    x_data = data[processing_util['training_columns']]
    y_data = data[processing_util['target_column']]    

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Custom Evaluation
def mape_eval(preds, dtrain):
    y_true = dtrain.get_label()
    # Avoid division by zero
    mask = y_true != 0
    y_true_safe = y_true[mask]
    preds_safe = preds[mask]
    
    mape = np.mean(np.abs((y_true_safe - preds_safe) / y_true_safe)) * 100
    # Return only name and value (not is_higher_better)
    return 'MAPE', mape


# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    param = {
        'objective': 'reg:absoluteerror',
        # 'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }
    
    # Parameters specific to tree-based models
    if param['booster'] in ['gbtree', 'dart']:
        param.update({
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        })
    
    # Parameters specific to dart booster
    if param['booster'] == 'dart':
        param.update({
            'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
            'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
            'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
        })
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # Train XGBoost model
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-mae')
    
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dvalid, 'validation')],
        early_stopping_rounds=500,
        custom_metric=mape_eval,
        # early_stopping_rounds=500,
        verbose_eval=10,
    )
    
    # Get best score
    best_score = model.best_score
    
    return best_score

# Main function to run Optuna study
def run_optuna_study(n_trials=100):
    # Create Optuna study
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction='minimize'
    )
    # def print_callback(study, trial):
    #     print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Best RMSE: {best_score:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    return best_params

# Function to test the model with best parameters
def test_best_model(best_params):
    # Create final model with best parameters
    final_model = xgb.XGBRegressor(**best_params)
    
    # Train on training data
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    # Make predictions on test data
    y_pred = final_model.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return final_model

if __name__ == "__main__":
    exp_name  = "exp_v0"
    data_dir = PROCESSED_DATA_DIR / exp_name
    # Load and split data
    X_train, X_test, y_train, y_test = load_data(data_dir=data_dir)
    
    # Create validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Run Optuna study
    print("Starting hyperparameter optimization...")
    best_params = run_optuna_study(n_trials=10)
    
    # Test best model
    print("\nTraining model with best parameters...")
    best_model = test_best_model(best_params)
    
    # Feature importance (if tree-based model)
    if best_params.get('booster', 'gbtree') in ['gbtree', 'dart']:
        print("\nFeature Importance:")
        importance = best_model.feature_importances_
        for i, imp in enumerate(importance):
            print(f"Feature {i}: {imp:.4f}")