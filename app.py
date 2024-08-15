from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

app = Flask(__name__)

STOCK_API_KEY = 'FVOEWU64HKN1C9U2'
STOCK_BASE_URL = 'https://www.alphavantage.co/query'
EXCHANGE_API_KEY = '699cc6dc962264a2fc44c056'
EXCHANGE_BASE_URL = 'https://v6.exchangerate-api.com/v6'
HOLIDAY_API_KEY = '0acd3ee9-301f-44ba-91b9-046154e1c874'
HOLIDAY_API_URL = 'https://holidayapi.com/v1/holidays'

def get_usd_to_inr_rate():
    url = f'{EXCHANGE_BASE_URL}/{EXCHANGE_API_KEY}/latest/USD'
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200 and 'conversion_rates' in data:
        return data['conversion_rates'].get('INR', 1)
    else:
        return 1  # If API fails, assume 1 USD = 1 INR (which will not be accurate)

def check_holiday(date):
    """
    This function checks if the given date is a holiday using the Holiday API.
    """
    params = {
        'api_key': HOLIDAY_API_KEY,
        'country': 'IN',  # Assuming we're checking holidays in India
        'year': date.year,
        'month': date.month,
        'day': date.day,
    }

    response = requests.get(HOLIDAY_API_URL, params=params)
    data = response.json()

    if 'holidays' in data and len(data['holidays']) > 0:
        # There's at least one holiday on this date
        holiday_name = data['holidays'][0]['name']
        return True, holiday_name
    else:
        # No holiday on this date
        return False, None

def is_holiday(date, country='IN'):
    url = f'{HOLIDAY_API_URL}'
    params = {
        'key': HOLIDAY_API_KEY,
        'country': country,
        'year': date.year,
        'month': date.month,
        'day': date.day
    }
    response = requests.get(url, params=params)
    data = response.json()
    if response.status_code == 200 and data.get('holidays'):
        return True
    return False

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df

def add_technical_indicators(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['High'] - df['Low']
    df['RSI'] = calculate_rsi(df)
    df = calculate_macd(df)
    df.fillna(0, inplace=True)
    return df

def train_random_forest(df):
    df = add_technical_indicators(df)

    # Features and target variable
    X = df[['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
    y = df['Close'].shift(-1)  # Predict the next day's closing price

    # Remove the last row with NaN target
    X = X[:-1]
    y = y[:-1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler

def fetch_stock_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': STOCK_API_KEY
    }

    response = requests.get(STOCK_BASE_URL, params=params)
    data = response.json()

    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)

        # Convert USD to INR
        usd_to_inr_rate = get_usd_to_inr_rate()
        df[['Open', 'High', 'Low', 'Close']] *= usd_to_inr_rate

        return df

    return None

def plot_prices(dates, predicted_prices, actual_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predicted_prices, label='Predicted Prices (INR)', color='blue', marker='o')
    plt.plot(dates, actual_prices, label='Actual Prices (INR)', color='green', marker='x')
    plt.title('Stock Prices in INR: Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plot_filename = 'static/plot.png'
    if os.path.exists(plot_filename):
        os.remove(plot_filename)  # Remove the old plot if it exists
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_prices = []
    actual_prices = []
    error_message = None
    future_dates = []
    future_prediction = None
    accuracy_score = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        future_date_str = request.form.get('future_date', '')

        # Fetch the stock data from Alpha Vantage
        df = fetch_stock_data(symbol)

        if df is not None:
            # Check if the future date is a holiday
            if future_date_str:
                future_date = pd.to_datetime(future_date_str)
                is_holiday, holiday_name = check_holiday(future_date)

                if is_holiday:
                    error_message = f"The date {future_date_str} is a holiday ({holiday_name}). No prediction will be made."
                    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                                           accuracy_score=accuracy_score)

            # Train the model
            model, scaler = train_random_forest(df)

            if future_date_str:
                future_date = pd.to_datetime(future_date_str)
                last_date = df.index[-1]

                if future_date > last_date:
                    # Prepare data for future date prediction
                    last_row = df.iloc[-1][['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
                    last_row_df = pd.DataFrame([last_row])
                    last_row_scaled = scaler.transform(last_row_df)

                    # Predict the price for the future date
                    future_prediction = model.predict(last_row_scaled)[0]

            # Generate dates from January 1st to today
            start_date = datetime(2024, 1, 1)
            end_date = datetime.now()
            date_range = pd.date_range(start=start_date, end=end_date)

            # Prepare data for predictions
            for date in date_range:
                if date in df.index:
                    last_row = df.loc[date][['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
                    last_row_df = pd.DataFrame([last_row])
                    last_row_scaled = scaler.transform(last_row_df)

                    # Predict the next day's price
                    predicted_price = model.predict(last_row_scaled)[0]
                    predicted_prices.append(predicted_price)
                    actual_prices.append(df.loc[date]['Close'])
                    future_dates.append(date)

            # Calculate accuracy score
            accuracy_score = r2_score(actual_prices, predicted_prices)*100

            # Plot the prices
            plot_filename = plot_prices(future_dates, predicted_prices, actual_prices)

            return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                   future_dates=future_dates, plot_url=plot_filename, future_prediction=future_prediction,
                                   accuracy_score=accuracy_score)

        else:
            error_message = "Failed to fetch stock data. Please check the stock symbol."

    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                           accuracy_score=accuracy_score)

if __name__ == '__main__':
    app.run(debug=True)
