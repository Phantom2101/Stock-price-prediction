from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Importing mdates for date formatting
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor  # Importing RandomForestRegressor

app = Flask(__name__)

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date='1990-01-01'):
    stock_data = yf.download(symbol, start=start_date)
    if stock_data.empty:
        print(f"No data found for {symbol}.")
        return None
    return stock_data

# Function to analyze stock performance
def analyze_stock_performance(data):
    if len(data) < 2:
        return "Not enough data to analyze.", "N/A"

    # Calculate percentage change
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    percentage_change = ((end_price - start_price) / start_price) * 100

    # Determine severity of change
    if percentage_change > 20:
        trend = "rapidly increased"
    elif percentage_change > 10:
        trend = "moderately increased"
    elif percentage_change > 0:
        trend = "slowly increased"
    elif percentage_change < 0:
        trend = "decreased"
    else:
        trend = "no change"

    # Determine if it's long-term or short-term
    term = "long-term" if data.index[-1] - data.index[0] > pd.Timedelta(days=365) else "short-term"

    return trend, term

# Function to create and train the Random Forest model
def create_random_forest_model(data):
    data = data[['Close']].copy()
    data['Days'] = np.arange(len(data))
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(data[['Days']], data['Close'])
    return model

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle stock data predictions
@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock']
    market = request.form['market']
    start_date = '2000-01-01'
    time_interval = int(request.form['time_interval'])

    if market == "India":
        stock_symbol += ".NS"  # Add .NS for NSE

    stock_data = fetch_stock_data(stock_symbol, start_date)

    if stock_data is not None and len(stock_data) > 20:
        model = create_random_forest_model(stock_data)

        last_day_index = len(stock_data)
        future_days = np.array([[last_day_index + i] for i in range(1, time_interval + 1)])
        predicted_prices = model.predict(future_days)

        current_price = stock_data['Close'].iloc[-1]

        # Create and save the graph
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index[-5:], stock_data['Close'][-5:], label='Last 5 Days Prices', color='blue')
        plt.axhline(y=current_price, color='g', linestyle='--', label='Current Price')

        for i in range(time_interval):
            plt.scatter(stock_data.index[-1] + pd.DateOffset(days=i + 1), predicted_prices[i],
                        label=f'Predicted Price (Day {i + 1})', zorder=5)

        plt.title(f"{stock_symbol} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price (INR/USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.gcf().autofmt_xdate()

        graph_path = os.path.join(app.root_path, 'static', 'graph.png')
        plt.savefig(graph_path)
        plt.close()

        # Pass the current date to the template
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('result.html',
                               stock_symbol=stock_symbol,
                               current_price=current_price,
                               predicted_prices=predicted_prices,
                               time_interval=time_interval,
                               predicted_trend="rise" if predicted_prices[-1] > current_price else "fall",
                               graph='graph.png',
                               current_date=current_date)  # Pass current date here
    else:
        return "Error fetching data. Please check the ticker symbol and try again."

# Route to display historical data
@app.route('/historical_data', methods=['GET'])
def historical_data():
    current_year = datetime.now().year  # Get the current year
    return render_template('historical_data.html', current_year=current_year)

# Route to display yearly stock data
@app.route('/display_historical_data', methods=['POST'])
def display_historical_data():
    stock_symbol = request.form['stock_symbol']
    year = request.form['year']
    market = request.form['market']
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Adjust the symbol for Indian stocks
    if market == "India":
        stock_symbol += ".NS"

    # Fetch historical data from Yahoo Finance
    stock_data = fetch_stock_data(stock_symbol, start_date)

    if stock_data is not None and not stock_data.empty:
        # Calculate the percentage change over the year
        first_price = stock_data['Close'].iloc[0]
        last_price = stock_data['Close'].iloc[-1]
        price_change = last_price - first_price
        percentage_change = (price_change / first_price) * 100

        # Determine the trend severity
        if price_change > 0:
            trend = "increased"
            if percentage_change > 20:
                severity = "rapidly"
            elif percentage_change > 10:
                severity = "moderately"
            else:
                severity = "slowly"
        else:
            trend = "decreased"
            if abs(percentage_change) > 20:
                severity = "rapidly "
            elif abs(percentage_change) > 10:
                severity = "moderately"
            else:
                severity = "slowly"

        # Determine if the stock is long-term or short-term
        if percentage_change > 0:
            stock_duration = "long-term" if percentage_change > 10 else "short-term"
        else:
            stock_duration = "long-term" if abs(percentage_change) > 10 else "short-term"

        # Create and save the yearly graph
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
        plt.title(f"{stock_symbol} Stock Prices in {year}")
        plt.xlabel("Date")
        plt.ylabel("Price (INR/USD)")
        plt.xticks(rotation=45)
        plt.legend()

        # Save the graph in the static folder
        graph_path = os.path.join(app.root_path, 'static', f'{stock_symbol}_{year}_graph.png')
        plt.savefig(graph_path)
        plt.close()

        return render_template('yearly_graph.html',
                               stock_symbol=stock_symbol,
                               year=year,
                               graph=f'{stock_symbol}_{year}_graph.png',
                               trend=trend,
                               severity=severity,
                               stock_duration=stock_duration)
    else:
        return "Error fetching data for the selected year. Please check the ticker symbol and try again."


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
