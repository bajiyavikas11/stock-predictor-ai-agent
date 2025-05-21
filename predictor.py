import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time
import torch
import torch.nn as nn
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from transformers import pipeline
from tqdm import tqdm
from torch.optim import Adam

def run_prediction(stock_symbol):
    stocks_symbols = [stock_symbol] if isinstance(stock_symbol, str) else stock_symbol
    shares_outstanding = {}
    for symbol in stocks_symbols:
        try:
            info = yf.Ticker(symbol).info
            shares_outstanding[symbol] = info.get('sharesOutstanding', None)
        except:
            shares_outstanding[symbol] = None

    data = yf.download(stocks_symbols, period="2mo", auto_adjust=False, group_by='ticker')
    all_data = []
    for symbol in stocks_symbols:
        df = data[symbol].copy()
        df['Ticker'] = symbol
        df.reset_index(inplace=True)
        shares = shares_outstanding[symbol]
        df['Market_Cap'] = df['Close'] * shares if shares is not None else None
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market_Cap']]
    total_market_cap_df = final_df.groupby('Date')['Market_Cap'].sum().reset_index()
    total_market_cap_df.rename(columns={'Market_Cap': 'Total_Market_Cap'}, inplace=True)
    final_df = pd.merge(final_df, total_market_cap_df, on='Date')
    final_df['Date'] = pd.to_datetime(final_df['Date'])

    #FINNHUB_API_KEY = api_key

    def fetch_finnhub_news(ticker, date):
        date = pd.to_datetime(date)
        from_str = date.strftime("%Y-%m-%d")
        to_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": from_str, "to": to_str, "token": "FINHUB API KEY"}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            return [{"Date": from_str, "Ticker": ticker, "Title": item["headline"]} for item in data[:5]]
        except Exception as e:
            print(f"Error: {ticker} | {date} → {e}")
            return []

    date_ticker_pairs = final_df[["Date", "Ticker"]].drop_duplicates()
    all_news = []
    print("Fetching news from Finnhub...")
    for _, row in tqdm(date_ticker_pairs.iterrows(), total=len(date_ticker_pairs)):
        news = fetch_finnhub_news(row["Ticker"], row["Date"])
        all_news.extend(news)
        time.sleep(1.1)

    news_data = pd.DataFrame(all_news)
    print("Fetched:", len(news_data), "articles")

    news_sentiment = news_data.copy()
    news_sentiment['Title_Rank'] = news_sentiment.groupby(['Date', 'Ticker']).cumcount()
    filtered_news = news_sentiment.pivot(index=['Date', 'Ticker'], columns='Title_Rank', values='Title')
    filtered_news.columns = [f"Top{i+1}" for i in range(filtered_news.shape[1])]
    filtered_news = filtered_news.reset_index()

    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", framework='pt')

    def get_sentiment_for_row(row):
        scores = []
        pos_count = neg_count = neu_count = 0
        headlines = [row.get(f"Top{i+1}") for i in range(5) if pd.notnull(row.get(f"Top{i+1}"))]
        for headline in headlines:
            result = sentiment_pipeline(headline)[0]
            label = result["label"].lower()
            score = result["score"]
            sentiment_val = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.2
            if sentiment_val == 1.0: pos_count += 1
            elif sentiment_val == -1.0: neg_count += 1
            else: neu_count += 1
            scores.append(sentiment_val * score)
        avg_sentiment = sum(scores) / len(scores) if scores else None
        return pd.Series({
            "Sentiment": avg_sentiment,
            "Positive_Count": pos_count,
            "Neutral_Count": neu_count,
            "Negative_Count": neg_count
        })

    sentiment_results = filtered_news.apply(get_sentiment_for_row, axis=1)
    filtered_news = pd.concat([filtered_news, sentiment_results], axis=1)
    filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
    final_df = pd.merge(final_df, filtered_news[['Date', 'Ticker', 'Sentiment']], on=['Date', 'Ticker'], how='left')

    features = ['Open', 'High', 'Low','Close', 'Volume', 'Market_Cap', 'Total_Market_Cap', 'Sentiment']
    data = final_df.copy()
    scaler = MinMaxScaler()
    scaler.fit(data[features])
    data_scaled = scaler.transform(data[features])

    SEQ_LEN = 30
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LEN):
        X.append(data_scaled[i:i+SEQ_LEN])
        y.append(data_scaled[i+SEQ_LEN, 3])

    X, y = np.array(X), np.array(y)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    # Mixed Attention LSTM Model
    class LSTMMixedAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMMixedAttention, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.feature_attn = nn.Linear(input_size, input_size)
            self.temporal_attn = nn.Linear(hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            feature_weights = torch.sigmoid(self.feature_attn(x))  # [batch, seq_len, input_size]
            x = x * feature_weights
            lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
            attn_weights = torch.softmax(self.temporal_attn(lstm_out), dim=1)  # [batch, seq_len, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_size]
            return self.fc(context)

    input_size = len(features)
    model = LSTMMixedAttention(input_size, 256, 4)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0005)

    percentage_errors = []
    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_preds = predictions.detach().numpy().flatten()
            batch_actuals = batch_y.detach().numpy().flatten()
            for pred, actual in zip(batch_preds, batch_actuals):
                if actual != 0:
                    percent_error = abs(pred - actual) / abs(actual) * 100
                    percentage_errors.append(percent_error)
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    average_percent_error = np.mean(percentage_errors)
    print(f"\nAverage Training Percentage Error (MAPE): {average_percent_error:.2f}%")

    def predict(model, data):
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(data)).numpy()


    last_day = X[-1].reshape(1, SEQ_LEN, input_size)
    predicted_price_scaled = predict(model, last_day)
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price_scaled[0][0], 0, 0, 0, 0]])[0][3]
    print(f"Predicted Next Close Price: {predicted_price:.2f}")
    return predicted_price
