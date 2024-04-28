# from iTransformer import iTransformer


# import pandas as pd

# # Path to the local CSV file
# file_path = './AirQualityUCI.csv'

# # Read the dataset from the local file
# data = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)

# # Remove rows where the Date or Time is missing
# data.dropna(subset=['Date', 'Time'], inplace=True)

# # Combine Date and Time into one column and convert to datetime
# data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

# # Set the new DateTime column as the index
# data.set_index('DateTime', inplace=True)

# # Drop the original Date and Time columns, as well as other irrelevant columns
# data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)

# # Filling missing values with interpolation
# data.interpolate(method='linear', limit_direction='both', inplace=True)

# # Preview the data
# print(data.head())

# # Now, data is ready for further analysis or modeling

# model = iTransformer()



from iTransformer import iTransformer
import pandas as pd


import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Path to the local CSV file
file_path = './AirQualityUCI.csv'

# Read the dataset from the local file
data = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)
data.dropna(subset=['Date', 'Time'], inplace=True)
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
data.set_index('DateTime', inplace=True)
data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)
data.interpolate(method='linear', limit_direction='both', inplace=True)

# Print the first few rows of the cleaned data


# need to order by date 
# make a data loader that will generate data for the itransformer 




class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback_len, prediction_len):
        self.data = data
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len

    def __len__(self):
        return len(self.data) - self.lookback_len - self.prediction_len + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.lookback_len]
        y = self.data[index + self.lookback_len:index + self.lookback_len + self.prediction_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Convert the DataFrame to a numpy array
data_np = data.to_numpy()
print(data_np.shape)
from sklearn.preprocessing import StandardScaler

# Assuming 'data_np' is your numpy array from the dataframe
scaler = StandardScaler()

# Fit the scaler to your data (this computes the mean and standard deviation)
scaler.fit(data_np)

# Transform the data using the fitted scaler (this applies the normalization)
data_normalized = scaler.transform(data_np)

# Now use this normalized data to create your dataset
dataset = TimeSeriesDataset(data_normalized, lookback_len=96, prediction_len=12)

# Assuming all columns after preprocessing are used as features
num_variates = data_np.shape[1]  # Number of features (columns) in your dataset

# Update the model initialization with the correct number of variates
model = iTransformer(
    num_variates=num_variates,
    lookback_len=96,
    depth=6,
    dim=64,
    num_tokens_per_variate=1,   # Depending on your approach, you might want to set more tokens per variate
    pred_length=12,              # Predicting one step ahead; adjust as needed for multiple step forecasting
    dim_head=32,
    heads=8,
    attn_dropout=0.1,
    ff_mult=4,
    ff_dropout=0.1
)

# Create the dataset
# dataset = TimeSeriesDataset(data_np, lookback_len=10, prediction_len=1)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(10000):  # Number of epochs
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        mse_loss = model(x_batch, targets=y_batch)
        mse_loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {mse_loss.item()}')

# Note: Include appropriate device management (CPU/GPU), checkpointing, and validation
