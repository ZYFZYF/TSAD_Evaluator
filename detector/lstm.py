# @Time    : 2022/3/8 15:34
# @Author  : ZYF
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from detector.detector import UnivariateDetector
from detector.fit import UnsupervisedFit
from detector.predict import OfflinePredict
from utils.utils import logging


class LSTMModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])


class LSTM(UnivariateDetector, UnsupervisedFit, OfflinePredict):
    def __init__(self, window_size, hidden_size=100, batch_size=32, epoch=100):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.epoch = epoch
        self.model = None
        self.optimizer = None

    def fit(self, x: np.ndarray):
        self.model = LSTMModel(hidden_size=self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        train_x = torch.tensor([x[i:i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor([x[i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        for epoch in range(self.epoch):
            dataset = TensorDataset(train_x, train_y)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            train_loss = []
            for ind, (value, next_value) in enumerate(data_loader):
                next_pred = self.model(value)
                loss = F.mse_loss(next_pred, next_value)
                self.optimizer.zero_grad()
                loss.backward()
                train_loss.append(loss.item())
                self.optimizer.step()
            logging.info(f"Epoch[{epoch}/{self.epoch}] Predict Loss: {sum(train_loss) / len(train_loss):.4f}")

    def predict(self, x: np.ndarray):
        self.model.eval()
        anomaly_scores = [0.0] * self.window_size
        train_x = torch.tensor([x[i:i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor([x[i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        dataset = TensorDataset(train_x, train_y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for ind, (value, next_value) in enumerate(data_loader):
            next_pred = self.model(value)
            for i in range(value.shape[0]):
                anomaly_scores.append(F.mse_loss(next_pred[i], next_value[i]).item())
        return anomaly_scores
