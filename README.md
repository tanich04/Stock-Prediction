# Stock Market Price Prediction with Sentiment Analysis
An end-to-end machine learning pipeline for predicting stock market trends by combining historical price data with news sentiment analysis.

## Project Overview
This project explores multiple approaches to stock price prediction, from traditional machine learning models to deep learning architectures, while incorporating sentiment analysis from financial news headlines to improve prediction accuracy.

### Key Features
Sentiment Analysis Integration: Uses transformer models (DistilBERT, RoBERTa) to extract sentiment scores from news headlines

Multiple Modeling Approaches: Linear Regression, Random Forest, Ensemble Methods, and Deep Learning (LSTM, GRU, BiLSTM)

Feature Engineering: Technical indicators (moving averages, Bollinger Bands) combined with sentiment features

Comprehensive Evaluation: RMSE-based model comparison and performance visualization

## Datasets
DJIA Dataset: Historical stock data (Open, High, Low, Close, Volume) for Dow Jones Industrial Average

News Headlines Dataset: Top 25 news headlines per date for sentiment analysis

Yahoo Finance News: Additional news data with titles and content for Apple stock analysis

## Methodology
1. Data Collection & Preprocessing
Stock data fetched using yfinance (Apple stock example in apple_webscrapping.py)

News headlines cleaned, combined, and preprocessed for sentiment analysis

Temporal alignment of stock and news data

2. Sentiment Analysis Pipeline
python
# Using DistilBERT for continuous sentiment scores
sentiment_model = pipeline("sentiment-analysis", 
                          model="distilbert-base-uncased-finetuned-sst-2-english",
                          return_all_scores=True)

# Using RoBERTa for multi-class sentiment (negative/neutral/positive)
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
3. Feature Engineering
Technical Indicators: Moving averages (5-day, 10-day), Exponential Moving Averages

Bollinger Bands: Upper/lower bands based on 20-day rolling standard deviation

Sentiment Features: Rolling sentiment averages (5-day, 10-day)

Lag Features: Previous day's closing price and volume

## Results
Model Performance Comparison
text
Model                                RMSE
------------------------------------------
Linear Regression (Baseline)         2.51
Random Forest                        1.83
Tuned Random Forest                   1.62
Voting Regressor                      1.58
Stacking Regressor                    1.55
Stacking + Feature Engineering        1.42
Stacking + Feature Selection          1.39
BiLSTM (3-day prediction)             2.87
Hybrid GRU-BiLSTM                     2.15
Key Insights
Sentiment Integration: Adding sentiment scores improved RMSE by ~15% compared to price-only models

Feature Engineering: Technical indicators and rolling statistics reduced error by another ~10%

Ensemble Methods: Stacking multiple models outperformed individual approaches

Deep Learning: Better at capturing temporal patterns but required more data and tuning

## Getting Started
Prerequisites
```bash
pip install -r requirements.txt
```
Main dependencies:

```pandas, numpy, scikit-learn

torch, transformers

yfinance, matplotlib

joblib
```

Quick Start
Collect Stock Data:

```bash
python apple_webscrapping.py
```
Run Sentiment Analysis:

```bash
# Use the Jupyter notebook or convert to script
jupyter notebook "Stock Sentiment Analysis.ipynb"
```
Train Models:

```bash
# For traditional ML approaches
python main2.ipynb
```

Transformer models: Hugging Face library

Financial data: Yahoo Finance via yfinance
