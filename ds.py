import pandas as pd

# Path to the local CSV file
file_path = './AirQualityUCI.csv'

# Read the dataset from the local file
data = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)

# Remove rows where the Date or Time is missing
data.dropna(subset=['Date', 'Time'], inplace=True)

# Combine Date and Time into one column and convert to datetime
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

# Set the new DateTime column as the index
data.set_index('DateTime', inplace=True)

# Drop the original Date and Time columns, as well as other irrelevant columns
data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)

# Filling missing values with interpolation
data.interpolate(method='linear', limit_direction='both', inplace=True)

# Preview the data
print(data.head())

# Now, data is ready for further analysis or modeling


# Example model initialization
model = iTransformer(
    num_variates=data.shape[1],       # Number of features (columns)
    lookback_len=24,                  # Number of past hours to consider for the forecast
    depth=6,                          # Number of layers in the model
    dim=64,                           # Dimension of the feature embeddings
    pred_length=1                     # Forecast horizon (e.g., predict the next hour)
)
