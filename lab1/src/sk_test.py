import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from typing import Tuple
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# 1.1
def data_preprocessing_regression(data_path: str, saved_to_disk: bool = False) -> Dataset:
    if saved_to_disk:
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
    
    def preprocess_function(examples):
        examples['Run_time'] = np.log(examples['Run_time'])
        return examples

    dataset = dataset.map(preprocess_function)
    return dataset

def data_split_regression(dataset: Dataset, test_size: float = 0.2, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = dataset['train'].to_pandas()
    X = df.drop(columns=['Run_time', '__index_level_0__']).values
    y = df['Run_time'].values
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

# 1.2
def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> SklearnLinearRegression:
    model = SklearnLinearRegression()
    model.fit(X_train, y_train)
    return model

# 1.3
def eval_linear_regression(model: SklearnLinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    relative_error = np.mean(np.abs(y_pred - y_test) / y_test)
    mean_pred = np.mean(y_pred)
    return mse, relative_error, mean_pred

# 2.1
def data_preprocessing_classification(data_path: str, mean: float, saved_to_disk: bool = False) -> Dataset:
    if saved_to_disk:
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
    
    def preprocess_function(examples):
        examples['Run_time'] = np.log(examples['Run_time'])
        return examples

    dataset = dataset.map(preprocess_function)
    
    df = dataset['train'].to_pandas()
    #df['label'] = (df['Run_time'] > mean).astype(int)
    #print(df['Run_time'])
    #df['label'] = (df['Run_time'] > mean).astype(int)
    df['Run_time'] = (df['Run_time'] > mean).astype(int)
    #print(df['Run_time'])
    #are_columns_equal = df['label'].equals(df['Run_time'])

    #print(f"Are 'label' and 'Run_time' columns equal? {are_columns_equal}")
    dataset = Dataset.from_pandas(df)
    return dataset

def data_split_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = dataset.to_pandas()
    X = df.drop(columns=['Run_time', '__index_level_0__'])
    #y = df['label']
    y = df['Run_time']
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test

# 2.2
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> SklearnLogisticRegression:
    model = SklearnLogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# 2.6
def eval_logistic_regression(model: SklearnLogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    # 回归任务
    regression_dataset = data_preprocessing_regression("Rosykunai/SGEMM_GPU_performance")
    X_train, X_test, y_train, y_test = data_split_regression(regression_dataset)
    regression_model = train_linear_regression(X_train, y_train)
    mse, relative_error, mean_pred = eval_linear_regression(regression_model, X_test, y_test)
    print(f"Regression MSE: {mse}, Relative Error: {relative_error}, Mean Prediction: {mean_pred}")

    # 分类任务
    classification_dataset = data_preprocessing_classification("Rosykunai/SGEMM_GPU_performance", mean=mean_pred)
    X_train, X_test, y_train, y_test = data_split_classification(classification_dataset)
    classification_model = train_logistic_regression(X_train, y_train)
    accuracy = eval_logistic_regression(classification_model, X_test, y_test)
    print(f"Classification Accuracy: {accuracy}")