# @Time    : 2022/3/10 14:56
# @Author  : ZYF
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import ANOMALY_SCORE_COLUMN
from detector.detector import UnivariateDetector
from detector.fit import UnsupervisedFit
from detector.predict import OfflinePredict
from utils.utils import logging


class MLPModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=input_size // 2)
        self.fc2 = torch.nn.Linear(in_features=input_size // 2, out_features=1)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MLP(UnivariateDetector, UnsupervisedFit, OfflinePredict):
    def __init__(self, window_size, batch_size=32, epoch=10, early_stop_epochs=3):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.epoch = epoch
        self.model = None
        self.optimizer = None
        self.model_params = None

    def save_model(self):
        self.model_params = self.model.state_dict()

    def load_model(self):
        self.model.load_state_dict(self.model_params)

    def fit(self, x: np.ndarray):
        self.model = MLPModel(input_size=self.window_size)
        optimizer = torch.optim.Adam(self.model.parameters())
        train_x = torch.tensor([x[i:i + self.window_size, 0].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor([x[i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        min_train_loss = np.inf
        not_update_round = 0
        for epoch in range(self.epoch):
            dataset = TensorDataset(train_x, train_y)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            train_loss = []
            for ind, (value, next_value) in enumerate(data_loader):
                next_pred = self.model(value)
                # print(value.shape, next_pred.shape, next_value.tolist(), next_pred.tolist())
                loss = F.mse_loss(next_pred, next_value)
                optimizer.zero_grad()
                loss.backward()
                train_loss.append(loss.item())
                optimizer.step()
            train_loss = sum(train_loss) / len(train_loss)
            logging.info(f"Epoch[{epoch}/{self.epoch}] Predict Loss: {train_loss:.4f}")
            if train_loss < min_train_loss:
                not_update_round = 0
                min_train_loss = train_loss
                self.save_model()
            else:
                not_update_round += 1
            if not_update_round == self.early_stop_epochs:
                break

    def predict(self, x: np.ndarray):
        self.load_model()
        self.model.eval()
        result = [{ANOMALY_SCORE_COLUMN: 0.0, 'predict_value': x[i, 0]} for i in range(self.window_size)]
        train_x = torch.tensor([x[i:i + self.window_size, 0].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor([x[i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        dataset = TensorDataset(train_x, train_y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for ind, (value, next_value) in enumerate(data_loader):
            next_pred = self.model(value)
            for i in range(value.shape[0]):
                # print(i, value[i].tolist())
                # print(next_pred[i], next_value[i], F.mse_loss(next_pred[i], next_value[i]),
                #       abs((next_pred[i] - next_value[i]).item()))
                result.append({ANOMALY_SCORE_COLUMN: abs((next_pred[i] - next_value[i]).item()),
                               'predict_value': next_pred[i].item()})
        return result
