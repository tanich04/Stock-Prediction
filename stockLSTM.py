import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("NEWS_YAHOO_stock_prediction.csv")

# Ensure the 'Date' column is of datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Sort the dataframe by date
df = df.sort_values('Date')

# Select only the 'Close' prices
closing_prices = df['Close'].values

df.drop_duplicates(subset=['Date','Close'],inplace=True)
print(closing_prices)

# Scale data
mm = MinMaxScaler()
scaled_close = mm.fit_transform(closing_prices[..., None]).squeeze()

# Define sequence length
seq_len = 15

# Create sequences
X = []
y = []

for i in range(len(scaled_close) - seq_len - 2): 
    X.append(scaled_close[i: i + seq_len])
    y.append(scaled_close[i + seq_len: i + seq_len + 3]) 

X = np.array(X)
y = np.array(y)

print(df['Close'].isnull())

# Split data into training and testing sets
train_size = int(0.8 * X.shape[0])
train_x = torch.from_numpy(X[:train_size]).float().unsqueeze(-1)
train_y = torch.from_numpy(y[:train_size]).float()
test_x = torch.from_numpy(X[train_size:]).float().unsqueeze(-1)
test_y = torch.from_numpy(y[train_size:]).float()

# Define Bidirectional LSTM model
class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 3)  # Predicting 3 future days

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate the hidden states from both directions
        return self.fc(hidden)

model = StockPredictionModel(input_size=1, hidden_size=64)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    output = model(train_x)
    loss = loss_fn(output, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    output = model(test_x)

# Inverse transform predictions and real values
pred = mm.inverse_transform(output.numpy().reshape(-1, 1)).reshape(-1, 3)
real = mm.inverse_transform(test_y.numpy().reshape(-1, 1)).reshape(-1, 3)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(real, pred))
print(f'RMSE: {rmse}')

# # Plot the results
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.plot(pred[:, i], color='red', label=f'Predicted Day {i + 1}' if i == 0 else "")
    plt.plot(real[:, i], color='green', label=f'Real Day {i + 1}' if i == 0 else "")
plt.legend()
plt.show()
