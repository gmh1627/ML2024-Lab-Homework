import yaml
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from typing import Tuple
from model import BaseModel

from utils import(
    TrainConfigR,
    TrainConfigC,
    DataLoader,
    Parameter,
    Loss,
    SGD,
    GD,
    save,
)

import pandas as pd

# You can add more imports if needed

# 1.1
def data_preprocessing_regression(data_path: str, saved_to_disk: bool = False) -> Dataset:
    r"""Load and preprocess the training data for the regression task.
    
    Args:
        data_path (str): The path to the training data.If you are using a dataset saved with save_to_disk(), you can use load_from_disk() to load the dataset.

    Returns:
        dataset (Dataset): The preprocessed dataset.
    """
    # 1.1-a
    # Load the dataset. Use load_from_disk() if you are using a dataset saved with save_to_disk()
    if saved_to_disk:
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path) 
    # Preprocess the dataset
    # Use dataset.to_pandas() to convert the dataset to a pandas DataFrame if you are more comfortable with pandas
    # TODO：You must do something in 'Run_time' column, and you can also do other preprocessing steps
    # 将每个子集转换为 Pandas DataFrame
    df = dataset['train'].to_pandas() if 'train' in dataset else None

    # 计算 Run_time 列的以 2 为底的对数
    df['Run_time'] = np.log2(df['Run_time'])

    dataset = Dataset.from_pandas(df)
    return dataset
    # dataset = Dataset.from_pandas(dataset) # Convert the pandas DataFrame back to a dataset
    return NotImplementedError

def data_split_regression(dataset: Dataset, batch_size: int, shuffle: bool) -> Tuple[DataLoader]:
    r"""Split the dataset and make it ready for training.

    Args:
        dataset (Dataset): The preprocessed dataset.
        batch_size (int): The batch size for training.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        A tuple of DataLoader: You should determine the number of DataLoader according to the number of splits.
    """
    # 1.1-b
    # Split the dataset using dataset.train_test_split() or other methods
    # TODO: Split the dataset
    # 第一次分割，返回的是Dataset类型
    train_dataset, remaining_dataset = dataset.train_test_split(test_size=0.2, shuffle=shuffle).values()

    # 第二次分割
    val_dataset, test_dataset = remaining_dataset.train_test_split(test_size=0.5, shuffle=shuffle).values()

    # Create a DataLoader for each split
    # TODO: Create a DataLoader for each split

    # 合并训练集和验证集
    train_dataset_df = Dataset.to_pandas(train_dataset)
    val_dataset_df = Dataset.to_pandas(val_dataset)
    combined_dataset_df = pd.concat([train_dataset_df, val_dataset_df], ignore_index=True)
    combined_dataset = Dataset.from_pandas(combined_dataset_df)

    train_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        train=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        train=False
    )

    return train_loader, test_loader
    return NotImplementedError

# 1.2
class LinearRegression(BaseModel):
    r"""A simple linear regression model.

    This model takes an input shaped as [batch_size, in_features] and returns
    an output shaped as [batch_size, out_features].

    For each sample [1, in_features], the model computes the output as:
    
    .. math::
        y = xW + b

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Example::

        >>> from model import LinearRegression
        >>> # Define the model
        >>> model = LinearRegression(3, 1)
        >>> # Predict
        >>> x = np.random.randn(10, 3)
        >>> y = model(x)
        >>> # Save the model parameters
        >>> state_dict = model.state_dict()
        >>> save(state_dict, 'model.pkl')
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 1.2-a
        # Look up the definition of BaseModel and Parameter in the utils.py file, and use them to register the parameters
        # TODO: Register the parameters
        self.weight = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.zeros(out_features))

        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

    def register_parameter(self, name: str, parameter: 'Parameter'):
        super().__setattr__(name, parameter)
        self._parameters[name] = parameter
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        # 1.2-b
        # Implement the forward pass of the model
        # TODO: Implement the forward pass
        return np.dot(x, self.weight) + self.bias
        return NotImplementedError

# 1.3
class MSELoss(Loss):
    r"""Mean squared error loss.

    This loss computes the mean squared error between the predicted and true values.

    Methods:
        __call__: Compute the loss
        backward: Compute the gradients of the loss with respect to the parameters
    """
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        r"""Compute the mean squared error loss.
        
        Args:
            y_pred: The predicted values
            y_true: The true values

        Returns:
            The mean squared error loss
        """
        # 1.3-a
        # Compute the mean squared error loss. Make sure y_pred and y_true have the same shape
        # TODO: Compute the mean squared error loss

        loss = 1/2 * np.mean((y_true - y_pred) ** 2)

        return loss
        return NotImplementedError
    
    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        r"""Compute the gradients of the loss with respect to the parameters.
        
        Args:
            x: The input values [batch_size, in_features]
            y_pred: The predicted values [batch_size, out_features]
            y_true: The true values [batch_size, out_features]

        Returns:
            The gradients of the loss with respect to the parameters, Dict[name, grad]
        """
        # 1.3-b
        # Make sure y_pred and y_true have the same shape
        # TODO: Compute the gradients of the loss with respect to the parameters
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        grad_weight = np.dot(x.T, (y_pred - y_true) / y_true.size)
        grad_bias = np.sum((y_pred - y_true) / y_true.size)
        grads = {'weight': grad_weight, 'bias': grad_bias}
        return grads
        return NotImplementedError
    

# 1.4
class TrainerR:
    r"""Trainer class to train for the regression task.

    Attributes:
        model (BaseModel): The model to be trained
        train_loader (DataLoader): The training data loader
        criterion (Loss): The loss function
        opt (SGD): The optimizer
        cfg (TrainConfigR): The configuration
        results_path (Path): The path to save the results
        step (int): The current optimization step
        train_num_steps (int): The total number of optimization steps
        checkpoint_path (Path): The path to save the model

    Methods:
        train: Train the model
        save_model: Save the model
    """
    def __init__(self, model: BaseModel, train_loader: DataLoader, loss: Loss, optimizer: SGD, config: TrainConfigR, results_path: Path):
        self.model = model
        self.train_loader = train_loader
        self.criterion = loss
        self.opt = optimizer
        self.cfg= config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps = len(self.train_loader) * self.cfg.epochs
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:
                # 1.4-a
                # load data from train_loader and compute the loss
                # TODO: Load data from train_loader and compute the loss
                for data_batch in self.train_loader:
                    features_batch = data_batch[:, :-1]  # 获取特征
                    targets_batch = data_batch[:, -1].reshape(-1, 1)    # 获取标签
                    
                    # 使用批量数据进行预测
                    y_pred_batch = self.model.predict(features_batch)
                    
                    # 计算损失
                    loss = self.criterion(y_pred_batch, targets_batch)
                    loss_list.append(loss)
                    
                    # Use pbar.set_description() to display current loss in the progress bar
                    pbar.set_description(f"Loss: {loss:.4f}, step:{self.step}")
                    # Compute the gradients of the loss with respect to the parameters
                    # Update the parameters with the gradients
                    # TODO: Compute gradients and update the parameters

                    grads = self.criterion.backward(features_batch, y_pred_batch, targets_batch)
                    self.opt.step(grads)
                    self.step += 1
                    pbar.update()

                    if self.step >= self.train_num_steps:
                        break
                if self.step >= self.train_num_steps:
                    break

        plt.plot(loss_list)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(self.results_path / 'loss_list.png')
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")
        
# 1.6
def eval_LinearRegression(model: LinearRegression, loader: DataLoader) -> Tuple[float,float]:
    r"""Evaluate the model on the given data.

    Args:
        model (LinearRegression): The model to evaluate.
        loader (DataLoader): The data to evaluate on.

    Returns:
        Tuple[float, float]: The average prediction, relative error.
    """
    model.eval()
    pred = np.array([])
    target = np.array([])
    # 1.6-a
    # Iterate over the data loader and compute the predictions
    # TODO: Evaluate the model

    # Compute the mean Run_time as Output
    # You can alse compute MSE and relative error
    # TODO: Compute metrics
    #print(f"Mean Squared Error: {mse}")

    #print(mu_target)

    #print(f"Relative Error: {relative_error}")

    return NotImplementedError


# 2.1
def data_preprocessing_classification(data_path: str, mean: float, saved_to_disk: bool = False) -> Dataset:
    r"""Load and preprocess the training data for the classification task.
    
    Args:
        data_path (str): The path to the training data.If you are using a dataset saved with save_to_disk(), you can use load_from_disk() to load the dataset.
        mean (float): The mean value to classify the data.

    Returns:
        dataset (Dataset): The preprocessed dataset.
    """
    # 2.1-a
    # Load the dataset. Use load_from_disk() if you are using a dataset saved with save_to_disk()
    if saved_to_disk:
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
    # Preprocess the dataset
    # Use dataset.to_pandas() to convert the dataset to a pandas DataFrame if you are more comfortable with pandas
    # TODO：You must do something in 'Run_time' column, and you can also do other preprocessing steps

    # dataset = Dataset.from_pandas(dataset) # Convert the pandas DataFrame back to a dataset
    return NotImplementedError

def data_split_classification(dataset: Dataset) -> Tuple[Dataset]:
    r"""Split the dataset and make it ready for training.

    Args:
        dataset (Dataset): The preprocessed dataset.

    Returns:
        A tuple of Dataset: You should determine the number of Dataset according to the number of splits.
    """
    # 2.1-b
    # Split the dataset using dataset.train_test_split() or other methods
    # TODO: Split the dataset

    return NotImplementedError

# 2.2
class LogisticRegression(BaseModel):
    r"""A simple logistic regression model for binary classification.

    This model takes an input shaped as [batch_size, in_features] and returns
    an output shaped as [batch_size, 1].

    For each sample [1, in_features], the model computes the output as:

    .. math::
        y = \sigma(xW + b)

    where :math:`\sigma` is the sigmoid function.

    .. Note::
        The model outputs the probability of the input belonging to class 1.
        You should use a threshold to convert the probability to a class label.

    Args:
        in_features (int): Number of input features.

    Example::
    
            >>> from model import LogisticRegression
            >>> # Define the model
            >>> model = LogisticRegression(3)
            >>> # Predict
            >>> x = np.random.randn(10, 3)
            >>> y = model(x)
            >>> # Save the model parameters
            >>> state_dict = model.state_dict()
            >>> save(state_dict, 'model.pkl')
    """
    def __init__(self, in_features: int):
        super().__init__()
        # 2.2-a
        # Look up the definition of BaseModel and Parameter in the utils.py file, and use them to register the parameters
        # This time, you should combine the weights and bias into a single parameter
        # TODO: Register the parameters


    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""Predict the probability of the input belonging to class 1.

        Args:
            x: The input values [batch_size, in_features]

        Returns:
            The probability of the input belonging to class 1 [batch_size, 1]
        """
        # 2.2-b
        # Implement the forward pass of the model
        # TODO: Implement the forward pass
        return NotImplementedError
    
# 2.3
class BCELoss(Loss):
    r"""Binary cross entropy loss.

    This loss computes the binary cross entropy loss between the predicted and true values.

    Methods:
        __call__: Compute the loss
        backward: Compute the gradients of the loss with respect to the parameters
    """
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        r"""Compute the binary cross entropy loss.
        
        Args:
            y_pred: The predicted values
            y_true: The true values

        Returns:
            The binary cross entropy loss
        """
        # 2.3-a
        # Compute the binary cross entropy loss. Make sure y_pred and y_true have the same shape
        # TODO: Compute the binary cross entropy loss
        return NotImplementedError
    
    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        r"""Compute the gradients of the loss with respect to the parameters.
        
        Args:
            x: The input values [batch_size, in_features]
            y_pred: The predicted values [batch_size, out_features]
            y_true: The true values [batch_size, out_features]

        Returns:
            The gradients of the loss with respect to the parameters [Dict[name, grad]]
        """
        # 2.3-b
        # Make sure y_pred and y_true have the same shape
        # TODO: Compute the gradients of the loss with respect to the parameters

        return NotImplementedError
    
# 2.4
class TrainerC:
    r"""Trainer class to train a model.

    Args:
        model (BaseModel): The model to train
        train_loader (DataLoader): The training data loader
        loss (Loss): The loss function
        optimizer (SGD): The optimizer
        config (dict): The configuration
        results_path (Path): The path to save the results
    """
    def __init__(self, model: BaseModel, dataset: np.ndarray, loss: Loss, optimizer: GD, config: TrainConfigC, results_path: Path):
        self.model = model
        self.dataset = dataset
        self.criterion = loss
        self.opt = optimizer
        self.cfg= config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps =  self.cfg.steps
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:
                # 2.4-a
                # load data from train_loader and compute the loss
                # TODO: Load data from train_loader and compute the loss

                # Use pbar.set_description() to display current loss in the progress bar

                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                # TODO: Compute gradients and update the parameters

                self.step += 1
                pbar.update()

        with open(self.results_path / 'loss_list.txt', 'w') as f:
            print(loss_list, file=f)
        plt.plot(loss_list)
        plt.savefig(self.results_path / 'loss_list.png')
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")

# 2.6
def eval_LogisticRegression(model: LogisticRegression, dataset: np.ndarray) -> float:
    r"""Evaluate the model on the given data.

    Args:
        model (LogisticRegression): The model to evaluate.
        dataset (np.ndarray): Test data

    Returns:
        float: The accuracy.
    """
    model.eval()
    correct = 0
    # 2.6-a
    # Iterate over the data and compute the accuracy
    # This time, we use the whole dataset instead of a DataLoader.Don't forget to add a bias term to the input
    # TODO: Evaluate the model

    return NotImplementedError