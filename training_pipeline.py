import pandas as pd
import os
import numpy as np
import yaml
import pickle
import mlflow
import xgboost as xgb
import optuna 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from mlflow.models.signature import infer_signature
from utils import model_eval, f1_eval
from prefect import task, flow, get_run_logger

from prefect_aws import S3Bucket

@task 
def load_config(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

@task
def download_s3_data(
    s3_bucket_block, 
    s3_folder_path, 
    dataset_folder_path
    ):
    s3_bucket_block = S3Bucket.load(s3_bucket_block)
    s3_bucket_block.download_folder_to_path(
        from_folder=s3_folder_path, 
        to_folder=dataset_folder_path,
        )

    return None

@task
def load_data(file_path):

    df = pd.read_csv(file_path)

    return df

@task
def data_split(
    df, 
    train_path, 
    valid_path, 
    test_path,
    ):

    train, test_val = train_test_split(df, test_size=0.25, random_state=42)
    valid, test = train_test_split(test_val, test_size=0.5, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)
    test.to_parquet(test_path, index=False)

    return None

@task
def process_features(
    train_path, 
    valid_path, 
    target_var, 
    save_dv=True
    ):

    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)

    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]
    x_valid = valid.drop(target_var, axis=1)
    y_valid = valid[target_var]

    dv = DictVectorizer()
    train_dict = x_train.to_dict(orient="records")
    val_dict = x_valid.to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)
    X_valid = dv.transform(val_dict)

    save_dv = True 

    if save_dv:
        os.makedirs("model", exist_ok=True)
        with open("model/preprocessor.b", "wb") as f:
            pickle.dump(dv, f)

    return X_train, y_train, X_valid, y_valid

@task
def hpo(
    X_train,
    y_train,
    X_valid, 
    y_valid
    ):
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    def objective(trial):
        
        with mlflow.start_run():
            
            params = {
                "objective": "binary:logistic",
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eta": trial.suggest_float("eta", 0.01, 0.1),
                "lambda": trial.suggest_float("lambda", 0.0, 1.0),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            }
        
            mlflow.log_params(params)

            # Create xgboost Classifier with the hyperparameters
            model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=500,
                evals=[(dvalid, "validation")],
                maximize=True, 
                custom_metric=f1_eval, 
                early_stopping_rounds=10, 
                verbose_eval=500,
            )

            # Evaluate the model's performance on the validation set
            y_pred_prob = model.predict(dvalid)
            y_pred = (y_pred_prob > 0.5).astype(int)
            auc, f1, precision, recall = model_eval(y_valid, y_pred, y_pred_prob)
            
            metrics = {
                "auc": auc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }
            signature = infer_signature(X_valid, y_pred)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("model/preprocessor.b", artifact_path="artifact")
            mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
            
        return -f1

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    return None

@flow(name="training_pipeline")
def train(config_path):

    experiment_name = os.getenv("EXPERIMENT_NAME", "training-pipeline")

    mlflow.set_tracking_uri("sqlite:///backend.db")
    mlflow.set_experiment(experiment_name)

    logger = get_run_logger()

    logger.info("Loading configuration")
    config = load_config(config_path)

    s3_folder_path = config["s3_folder_path"]
    dataset_folder_path = config["dataset_folder_path"]
    dataset_path = config["dataset_path"]
    train_path = config["train_path"]
    valid_path = config["valid_path"]
    test_path = config["test_path"]
    target_var = config["target_variable"]

    logger.info("Downloading the data from s3")
    download_s3_data("s3-final-pj", s3_folder_path, dataset_folder_path)

    logger.info("Loading the data")
    df = load_data(dataset_path)

    logger.info("Splitting the data into train, valid and test")
    data_split(df, train_path, valid_path, test_path)

    logger.info("Processing Features")
    X_train, y_train, X_valid, y_valid = process_features(train_path, valid_path, target_var)
    
    logger.info("Hyperparameter Tuning with XGBoost model")
    hpo(X_train, y_train, X_valid, y_valid)

if __name__ == "__main__":

    train("config.yaml")

    





















