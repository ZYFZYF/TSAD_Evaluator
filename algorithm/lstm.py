# @Time    : 2022/3/8 15:34
# @Author  : ZYF
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from config import ANOMALY_SCORE_COLUMN
from detector.detector import UnivariateDetector
from detector.fit import UnsupervisedFit
from detector.predict import OfflinePredict
from utils.pt import describe_torch_model
from utils.utils import logging


class LSTMModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output[:, -1:, :])


class LSTM(UnivariateDetector, UnsupervisedFit, OfflinePredict):
    def __init__(self, window_size, hidden_size=100, batch_size=32, epoch=10, early_stop_epochs=3):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.early_stop_epochs = early_stop_epochs
        self.epoch = epoch
        self.model = None
        self.optimizer = None
        self.model_params = None
        self.scaler = MinMaxScaler()

    def save_model(self):
        self.model_params = self.model.state_dict()

    def load_model(self):
        self.model.load_state_dict(self.model_params)

    def fit(self, x: np.ndarray):
        self.model = LSTMModel(hidden_size=self.hidden_size)
        x = self.scaler.fit_transform(x)
        describe_torch_model(self.model)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.MSELoss()
        train_x = torch.tensor([x[i:i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor(
            [x[i + self.window_size:i + self.window_size + 1].tolist() for i in range(x.shape[0] - self.window_size)])
        min_train_loss = np.inf
        not_update_round = 0
        for epoch in range(self.epoch):
            dataset = TensorDataset(train_x, train_y)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            train_loss = []
            for ind, (value, next_value) in enumerate(data_loader):
                next_pred = self.model(value)
                loss = criterion(next_pred, next_value)
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
        x = self.scaler.transform(x)
        train_x = torch.tensor([x[i:i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        train_y = torch.tensor([x[i + self.window_size].tolist() for i in range(x.shape[0] - self.window_size)])
        dataset = TensorDataset(train_x, train_y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for ind, (value, next_value) in enumerate(data_loader):
            next_pred = self.model(value)
            for i in range(value.shape[0]):
                predict = self.scaler.inverse_transform([[next_pred[i].item()]])[0][0]
                label = self.scaler.inverse_transform([[next_value[i].item()]])[0][0]
                # print(next_pred[i].item(), predict, abs((predict - label).item()))
                result.append({ANOMALY_SCORE_COLUMN: abs(predict - label),
                               'predict_value': predict})
        return result
