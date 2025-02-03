import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.io as pio
import h5py

# Create output directory
output_dir = './swe_outputs'
os.makedirs(output_dir, exist_ok=True)

# Redirect output to both terminal and a log file
class DualOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

import sys
sys.stdout = DualOutput(os.path.join(output_dir, 'output_log.txt'))

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load single CSV file
data_path = './processed_snotel_data_filled 1.csv'
swe_data = pd.read_csv(data_path)

# Handling missing values
swe_data.fillna(method='ffill', inplace=True)

# Filter for Winter Season (December 1 to May 31)
swe_data['date'] = pd.to_datetime(swe_data['date'])
swe_data = swe_data[(swe_data['date'].dt.month >= 12) | (swe_data['date'].dt.month <= 5)]
swe_data = swe_data[(swe_data['date'].dt.year >= 2008) | (swe_data['date'].dt.year <= 2016)]

# Feature Engineering
swe_data['day_of_year'] = swe_data['date'].dt.dayofyear
swe_data['month'] = swe_data['date'].dt.month

# Features and Target
features = ['latitude', 'longitude', 'elevation', 'day_of_year', 'month', 
            'precip', 'tmin', 'tmax', 'SPH', 'SRAD', 'Rmax', 'Rmin', 'windspeed', 'southness']
target = 'SWE'

# Normalizing features
scaler = StandardScaler()
swe_data[features] = scaler.fit_transform(swe_data[features])

# Splitting data
train_data = swe_data.sample(frac=0.8, random_state=42)
test_data = swe_data.drop(train_data.index)

train_input = train_data[features].values.reshape(-1, 1, len(features))
train_output = train_data[target].values.reshape(-1, 1, 1)

test_input = test_data[features].values.reshape(-1, 1, len(features))
test_output = test_data[target].values.reshape(-1, 1, 1)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.25, max_len: int = 1000000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1).to(device=device)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device=device)
        pe = torch.zeros(max_len, 1, d_model).to(device=device)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

# SWE Prediction Model
class SWEModel(nn.Module):
    def __init__(self, input_dim=14, model_dim=128, seq_length=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, dropout=0.2), num_layers=6
        )
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = x.to(device)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc_out(x)

# Model, Loss, Optimizer
model = SWEModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training with Mini-Batches
batch_size = 64
num_epochs = 30

train_losses = []
rmse_list = []
r2_list = []
nse_list = []
relative_bias_list = []
actual_error_list = []

model.train()
for epoch in range(num_epochs):
    permutation = np.random.permutation(train_input.shape[0])
    total_loss = 0

    for i in range(0, train_input.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_inputs = torch.from_numpy(train_input[indices]).float().to(device)
        batch_targets = torch.from_numpy(train_output[indices]).float().to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (train_input.shape[0]   / batch_size)
    train_losses.append(avg_loss)

    # Evaluation Metrics Per Epoch
    model.eval()
    with torch.no_grad():
        val_inputs = torch.from_numpy(test_input).float().to(device)
        val_targets = torch.from_numpy(test_output).float().to(device)
        predictions = model(val_inputs)

        mse = criterion(predictions, val_targets).item()
        rmse = math.sqrt(mse)
        r2 = r2_score(val_targets.cpu().numpy().flatten(), predictions.cpu().numpy().flatten())
        nse = 1 - (np.sum((val_targets.cpu().numpy().flatten() - predictions.cpu().numpy().flatten())**2) /
                   np.sum((val_targets.cpu().numpy().flatten() - np.mean(val_targets.cpu().numpy().flatten()))**2))
        relative_bias = (np.mean(predictions.cpu().numpy() - val_targets.cpu().numpy()) / np.mean(val_targets.cpu().numpy())) * 100
        actual_error = np.mean(np.abs(predictions.cpu().numpy() - val_targets.cpu().numpy()))

        rmse_list.append(rmse)
        r2_list.append(r2)
        nse_list.append(nse)
        relative_bias_list.append(relative_bias)
        actual_error_list.append(actual_error)

    print(f'Epoch {epoch + 1}, Loss: {avg_loss}, RMSE: {rmse}, R^2: {r2}')

# Save the trained model
model_save_path = os.path.join(output_dir, 'swe_model.h5')
with h5py.File(model_save_path, 'w') as f:
    for name, param in model.state_dict().items():
        f.create_dataset(name, data=param.cpu().numpy())
print(f'Model saved to {model_save_path}')

# Plotting Metrics
metrics = {'Loss': train_losses, 'RMSE': rmse_list, 'R2': r2_list, 'NSE': nse_list, 'Relative Bias': relative_bias_list, 'Actual Error': actual_error_list}
for metric, values in metrics.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), values, marker='o')
    plt.title(f'{metric} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_curve.png'))

# 3D US Map Example
sample_locations = swe_data.sample(100)
fig = px.scatter_3d(sample_locations, x='longitude', y='latitude', z='SWE', color='SWE')
pio.write_html(fig, os.path.join(output_dir, '3d_us_map.html'))
fig.write_image(os.path.join(output_dir, '3d_us_map.jpeg'))

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
