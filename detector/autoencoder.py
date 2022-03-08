# @Time    : 2022/3/8 11:42
# @Author  : ZYF
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from detector.detector import UnivariateDetector
from detector.fit import UnsupervisedFit
from detector.predict import OfflinePredict

logging.basicConfig(level=logging.INFO)


def distance(x, y):
    return torch.norm(x - y, p=2, dim=1)


class AutoEncoderModel(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, z_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, distance(x, x_hat)


class AutoEncoder(UnivariateDetector, UnsupervisedFit, OfflinePredict):

    def __init__(self, window_size, z_dim, batch_size=32, epoch=10, early_stop_epochs=3):
        super().__init__()
        self.optimizer = None
        self.window_size = window_size
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.max_epoch = epoch
        self.model = None
        self.model_params = None
        self.early_stop_epochs = early_stop_epochs

    def save_model(self):
        self.model_params = self.model.state_dict()

    def load_model(self):
        self.model.load_state_dict(self.model_params)

    def fit(self, x: np.ndarray):
        self.model = AutoEncoderModel(in_dim=self.window_size, z_dim=self.z_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        train_x = torch.tensor(
            [x[i:i + self.window_size, 0].tolist() for i in range(x.shape[0] - self.window_size + 1)])
        train_x = train_x
        train_x, val_x = train_test_split(train_x, shuffle=True)
        train_dataset = TensorDataset(train_x)
        val_dataset = TensorDataset(val_x)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        min_val_loss = np.inf
        not_update_round = 0
        for epoch in range(self.max_epoch):
            # train
            for value in train_data_loader:
                _, loss = self.model(value[0])
                self.optimizer.zero_grad()
                torch.mean(loss).backward()
                self.optimizer.step()
            # val
            val_loss = []
            for value in val_data_loader:
                _, loss = self.model(value[0])
                val_loss.append(torch.mean(loss))
            val_loss = sum(val_loss) / len(val_loss)
            logging.info(f'Epoch {epoch + 1}/{self.max_epoch}: val_loss = {val_loss:.5f}')
            # save
            if val_loss < min_val_loss:
                not_update_round = 0
                min_val_loss = val_loss
                self.save_model()
            else:
                not_update_round += 1
            # early stop
            if not_update_round == self.early_stop_epochs:
                break

    def predict(self, x: np.ndarray):
        self.load_model()
        self.model.eval()
        anomaly_scores = [0.0] * (self.window_size - 1)
        test_x = torch.tensor([x[i:i + self.window_size, 0].tolist() for i in range(x.shape[0] - self.window_size + 1)])
        for value in DataLoader(TensorDataset(test_x), batch_size=self.batch_size, shuffle=False):
            _, loss = self.model(value[0])
            for i in range(value[0].shape[0]):
                anomaly_scores.append(loss[i].item())
        return anomaly_scores
