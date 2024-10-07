import yaml
import dataclasses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from typing import Tuple
from model import BaseModel
from sklearn.preprocessing import StandardScaler
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
        #dataset = load_dataset("Rosykunai/SGEMM_GPU_performance")
        dataset = load_dataset(data_path)
    # Preprocess the dataset
    # Use dataset.to_pandas() to convert the dataset to a pandas DataFrame if you are more comfortable with pandas
    # TODO：You must do something in 'Run_time' column, and you can also do other preprocessing steps

    df = dataset['train'].to_pandas()
    df['Run_time'] = np.log(df['Run_time'])
    
    run_time = df['Run_time']
    features = df.drop(columns=['Run_time'])

    # Manually standardize the features (mean=0, std=1)
    features_standardized = (features - features.mean()) / features.std()

    # Combine the standardized features and Run_time column back into a DataFrame
    df_standardized = pd.DataFrame(features_standardized, columns=features.columns)
    df_standardized['Run_time'] = run_time.values
    dataset = Dataset.from_pandas(df_standardized)
    '''
    scaler = StandardScaler()

    # 获取需要标准化的列名
    columns_to_scale = df.columns[df.columns != 'Run_time']

    # 对这些列应用scaler
    for column in columns_to_scale:
        df[column] = scaler.fit_transform(df[[column]])

    dataset = Dataset.from_pandas(df)
    '''
    return dataset

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
    train_spilt = dataset.train_test_split(test_size=0.2, shuffle=shuffle)
    train_dataset = train_spilt['train']
    temp_dataset = train_spilt['test']

    val_spilt = temp_dataset.train_test_split(test_size=0.5, shuffle=shuffle)
    val_dataset = val_spilt['train']
    test_dataset = val_spilt['test']
    #train_dataset = Dataset.from_pandas(train_dataset.to_pandas().append(val_dataset.to_pandas(), ignore_index=True))
    # Create a DataLoader for each split
    # TODO: Create a DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size, shuffle, train=True)
    #test_loader = DataLoader(test_dataset, batch_size, shuffle, train=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle, train=False)
    return train_loader, val_loader

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
        self.W = Parameter(np.random.randn(in_features, out_features))
        self.b = Parameter(np.random.randn(out_features))
        #self.register_parameters(W=self.W, b=self.b)
    #     self.register_parameter('W', self.W)
    #     self.register_parameter('b', self.b)
        
    # def register_parameter(self, name: str, parameter: 'Parameter'):
    #     super().__setattr__(name, parameter)
    #     self._parameters[name] = parameter

    def predict(self, x: np.ndarray) -> np.ndarray:
        # 1.2-b
        # Implement the forward pass of the model
        # TODO: Implement the forward pass
        return x @ self.W + self.b

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
        mse = 0.5 * np.mean((y_pred - y_true) ** 2)
        return mse
    
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
        grad_W = (1 / x. shape[0]) * x.T @ (y_pred - y_true)
        grad_b = (1 / x.shape[0]) * np.sum(y_pred - y_true)
        return {"W": grad_W, "b": grad_b}
    
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
                for batch in self.train_loader:
                    features = batch[:, :-1]
                    target = batch[:, -1].reshape(-1, 1)
                    pred = self.model.predict(features)
                    loss = self.criterion(pred, target)
                    loss_list.append(loss.item())
                    
                    self.opt.step(self.criterion.backward(features, pred, target))
                    pbar.set_description(f"Loss: {loss:.4f}")
                    self.step += 1
                    pbar.update()
                    if self.step >= self.train_num_steps:
                        break
                # Use pbar.set_description() to display current loss in the progress bar

                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                # TODO: Compute gradients and update the parameters
        
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
    for batch in loader:
        features = batch[:, :-1]
        target_value = batch[:, -1]
        pred_batch = model.predict(features)
        pred = np.append(pred, pred_batch)
        target = np.append(target, target_value)
    # Compute the mean Run_time as Output
    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    # You can alse compute MSE and relative error
    mse = np.mean((pred - target) ** 2)
    #relative_error = np.abs(mu_pred - mu_target) / mu_target if mu_target != 0 else 0
    relative_error = np.abs(mu_pred - mu_target) / mu_target
    # TODO: Compute metrics
    print(f"Mean Target: {mu_target}")
    print(f"Mean Squared Error: {mse}")
    return mu_pred, relative_error

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
    df = dataset['train'].to_pandas()
    df['Run_time'] = np.log(df['Run_time'])
    df['label'] = (df['Run_time'] > mean).astype(int)
    #print(mean)
    #print(df['Run_time'])
    # Manually standardize the features (mean=0, std=1)
    label = df['label']
    #print(df['Run_time'])
    #print(label)
    features = df.drop(columns=['Run_time', 'label'])
    features_standardized = (features - features.mean()) / features.std()

    # Combine the standardized features and Run_time column back into a DataFrame
    df_standardized = pd.DataFrame(features_standardized, columns=features.columns)
    df_standardized['label'] = label.values
    #print(df_standardized)
    dataset = Dataset.from_pandas(df_standardized)
    return dataset

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
    train_split = dataset.train_test_split(test_size=0.2, shuffle=True)
    train_dataset = train_split['train']
    temp_dataset = train_split['test']

    # Split the temp dataset into val and test
    val_split = temp_dataset.train_test_split(test_size=0.5, shuffle=True)
    val_dataset = val_split['train']
    test_dataset = val_split['test']
    return train_dataset, val_dataset

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
        #print(f"in_features: {in_features}")
        self.beta = Parameter(np.random.randn(in_features + 1, 1))

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
        #x = np.hstack([x, np.ones((x.shape[0], 1))])# 将输入 x 扩展一列全为 1 的列，以便与 beta 相乘
        #print(f"x shape: {x.shape}")
        #print(f"beta shape: {self.beta.shape}")
        z = x @ self.beta
        return 1 / (1 + np.exp(-z))
    
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
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Clip the predicted values to avoid log(0)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
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
        #grad_y_pred = (y_pred - y_true) / (y_pred * (1 - y_pred) * y_pred.shape[0])
        grad_beta = x.T @ (y_pred  - y_true) / x.shape[0]
        return {"beta": grad_beta}
    
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
        x, y_true = self.dataset[:, :-1], self.dataset[:, -1].reshape(-1, 1)
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            prev_loss = float('inf')
            while self.step < self.train_num_steps:
                # 2.4-a
                # load data from train_loader and compute the loss
                # TODO: Load data from train_loader and compute the loss
                
                # Use pbar.set_description() to display current loss in the progress bar

                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                # TODO: Compute gradients and update the parameters
                y_pred = self.model.predict(x)
                loss = self.criterion(y_pred, y_true)
                loss_list.append(loss)
                pbar.set_description(f"Loss: {loss:.4f}")
                grad = self.criterion.backward(x, y_pred, y_true)
                self.opt.step(grad)
                if abs(prev_loss - loss) < 1e-6:
                    break
                prev_loss = loss

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
    x, y_true = dataset[:, :-1], dataset[:, -1].reshape(-1, 1)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    prob = model.predict(x)
    y_pred = (prob > 0.5).astype(int)
    correct = np.sum(y_pred == y_true)
    accuracy = correct / len(y_true)
    
    return accuracy