import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("AAPL_Closing_Prices.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Calculate moving average of the closing prices
df['Moving_Average'] = df['Close'].rolling(window=3).mean()
df.dropna(inplace=True)  # Drop the rows where any of the elements are NaN

# Feature scaling
mm = MinMaxScaler()
scaled_features = mm.fit_transform(df[['Close', 'Moving_Average']])
close_scaler = MinMaxScaler()  # Scaler for the Close column only
close_scaled = close_scaler.fit_transform(df[['Close']])

# Define sequence length and prepare data for training
seq_len = 3
X = []
y = []

for i in range(len(scaled_features) - seq_len):
    X.append(scaled_features[i:i + seq_len])
    y.append(close_scaled[i + seq_len, 0])  # Target is the scaled closing price of the next day

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
train_size = int(0.8 * len(X))
train_x = torch.tensor(X[:train_size], dtype=torch.float32)
train_y = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)
test_x = torch.tensor(X[train_size:], dtype=torch.float32)
test_y = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

# Define the model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("AAPL_Closing_Prices.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Calculate moving average of the closing prices
df['Moving_Average'] = df['Close'].rolling(window=3).mean()
df.dropna(inplace=True)  # Drop the rows where any of the elements are NaN

# Feature scaling
mm = MinMaxScaler()
scaled_features = mm.fit_transform(df[['Close', 'Moving_Average']])
close_scaler = MinMaxScaler()  # Scaler for the Close column only
close_scaled = close_scaler.fit_transform(df[['Close']])

# Define sequence length and prepare data for training
seq_len = 3
X = []
y = []

for i in range(len(scaled_features) - seq_len):
    X.append(scaled_features[i:i + seq_len])
    y.append(close_scaled[i + seq_len, 0])  # Target is the scaled closing price of the next day

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
train_size = int(0.8 * len(X))
train_x = torch.tensor(X[:train_size], dtype=torch.float32)
train_y = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)
test_x = torch.tensor(X[train_size:], dtype=torch.float32)
test_y = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

# Define the model
class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockPredictionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out, _ = self.bilstm(out)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out

# Initialize the model
model = StockPredictionModel(input_size=2, hidden_size=64)  # input_size = number of features

# Training configuration
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    output = model(train_x)
    loss = loss_fn(output, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    predicted = model(test_x)
    predicted = predicted.numpy()
    predicted = close_scaler.inverse_transform(predicted)  # Use the Close column scaler for inverse transformation
    actual = close_scaler.inverse_transform(test_y.numpy())  # Use the Close column scaler for inverse transformation

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(15, 5))
plt.plot(predicted, label='Predicted Prices')
plt.plot(actual, label='Actual Prices')
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()


# Initialize the model
model = StockPredictionModel(input_size=2, hidden_size=64)  # input_size = number of features

# Training configuration
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    output = model(train_x)
    loss = loss_fn(output, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    predicted = model(test_x)
    predicted = predicted.numpy()
    predicted = close_scaler.inverse_transform(predicted)  # Use the Close column scaler for inverse transformation
    actual = close_scaler.inverse_transform(test_y.numpy())  # Use the Close column scaler for inverse transformation

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(15, 5))
plt.plot(predicted, label='Predicted Prices')
plt.plot(actual, label='Actual Prices')
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
