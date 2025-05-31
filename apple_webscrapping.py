import yfinance as yf
import pandas as pd

# Define the ticker symbol
tickerSymbol = 'AAPL'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1y')  # Using "1y" to get the last year's data

# Select only the 'Close' column
closing_prices = tickerDf['Close']

# Save the data to a CSV file
closing_prices.to_csv('AAPL_Closing_Prices.csv')

print('Data saved to AAPL_Closing_Prices.csv')
