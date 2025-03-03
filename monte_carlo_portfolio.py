# -*- coding: utf-8 -*-
"""Monte_Carlo_Portfolio.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1smjJMDIF6yfCT9gY-QU3dyfOiU3SusMH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define parameters
stock_symbols = ['BRK-B', 'WFC', 'OKE', 'NUE', 'HII','MSTR','XOM','TSLA','GEO','XLV','XLK','XLI','XLF','XLE']
benchmark_symbol = '^GSPC'  # S&P 500 index
start_date = '1999-01-01'
end_date = '2024-12-31'
simulation_runs = 1000
time_horizon = 252  # 1 year of trading days

# Fetch historical data using Yahoo Finance API
all_symbols = stock_symbols + [benchmark_symbol]
stock_data = yf.download(all_symbols, start=start_date, end=end_date)
print("Stock data retrieved:")
print(stock_data.head())  # Debugging output

# Generate synthetic historical stock return data
np.random.seed(42)
dates = pd.date_range(start="1999-01-01", end="2024-12-31", freq="B")  # Business days
num_stocks = len(stock_symbols)

# Simulated daily log returns for each stock
historical_stock_returns = np.random.normal(loc=0.0005, scale=0.01, size=(len(dates), num_stocks))

# Convert to a DataFrame with stock symbols as columns
historical_stock_returns_df = pd.DataFrame(historical_stock_returns, index=dates, columns=stock_symbols)

# Compute cumulative returns for each stock
historical_cumulative_returns_stocks = (1 + historical_stock_returns_df).cumprod()

# Define moving average windows
short_window = 50  # 50-day moving average
long_window = 200  # 200-day moving average

# Calculate moving averages for each stock
short_moving_avg_stocks = historical_cumulative_returns_stocks.rolling(window=short_window).mean()
long_moving_avg_stocks = historical_cumulative_returns_stocks.rolling(window=long_window).mean()

# Plot cumulative returns and moving averages for each stock
fig, ax = plt.subplots(figsize=(12, 6))

for stock in stock_symbols:
    ax.plot(historical_cumulative_returns_stocks[stock], label=f"{stock} Cumulative Return", alpha=0.7)
    ax.plot(short_moving_avg_stocks[stock], linestyle="--", alpha=0.6, label=f"{stock} {short_window}-Day MA")
    ax.plot(long_moving_avg_stocks[stock], linestyle="--", alpha=0.6, label=f"{stock} {long_window}-Day MA")

# Labels and title
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Return")
ax.set_title("Stock Cumulative Return with Moving Averages")
ax.legend(loc="upper left", ncol=2, fontsize=8)
ax.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

# Compute key statistics for each stock
stock_stats_df = pd.DataFrame(index=stock_symbols)

# Calculate annualized return and volatility
annualized_return = historical_stock_returns_df.mean() * 252 * 100  # Convert to percentage
annualized_volatility = historical_stock_returns_df.std() * np.sqrt(252) * 100  # Convert to percentage

# Define risk-free rate (assume 2%)
risk_free_rate = 2

# Compute Sharpe Ratio
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Populate the DataFrame with statistics
stock_stats_df["Annualized Return (%)"] = annualized_return
stock_stats_df["Annualized Volatility (%)"] = annualized_volatility
stock_stats_df["Sharpe Ratio"] = sharpe_ratio

# Print the statistics table
print(stock_stats_df)

# Ensure data includes 'Adj Close' or fallback to 'Close'
if 'Adj Close' in stock_data:
    stock_data = stock_data['Adj Close']
else:
    stock_data = stock_data['Close']

# Check available columns
available_stocks = list(stock_data.columns)
valid_stocks = [stock for stock in stock_symbols if stock in available_stocks]
print("Valid stocks:", valid_stocks)  # Debugging output

# Filter stock data to valid stocks
stock_data = stock_data[valid_stocks + [benchmark_symbol]].dropna()
print("Stock data after NaN removal:")
print(stock_data.head())

# Calculate log returns
log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
mean_returns = log_returns.mean()
std_dev = log_returns.std()

# Define portfolio weights (equal weighting for now)
num_stocks = len(valid_stocks)
weights = np.array([1/num_stocks] * num_stocks) if num_stocks > 0 else np.array([])
print("Weights:", weights)  # Debugging output

# Ensure portfolio_cumulative is correctly calculated
if num_stocks > 0:
    print("Stock data shape:", stock_data.shape)
    print("Weights shape:", weights.shape)
    print("First row shape:", stock_data.iloc[0].shape)
    print("Initial portfolio value:", stock_data.iloc[0][valid_stocks] @ weights)

    try:
        portfolio_cumulative = (stock_data[valid_stocks] @ weights) / (stock_data.iloc[0][valid_stocks] @ weights)
        portfolio_cumulative = portfolio_cumulative.dropna()  # Drop NaN values just in case
        portfolio_cumulative.index = pd.to_datetime(portfolio_cumulative.index)  # Ensure proper datetime index
        print("Portfolio cumulative head:")
        print(portfolio_cumulative.head())
    except Exception as e:
        print("Error in portfolio calculation:", e)
        portfolio_cumulative = pd.Series(dtype=float)
else:
    print("No valid stocks in portfolio!")
    portfolio_cumulative = pd.Series(dtype=float)

# Debug: Check if portfolio_cumulative is empty
if portfolio_cumulative.empty:
    print("Warning: portfolio_cumulative is empty! No portfolio data available.")

# Ensure S&P 500 cumulative returns are correctly calculated
if benchmark_symbol in stock_data.columns:
    try:
        sp500_cumulative = (stock_data[benchmark_symbol] / stock_data[benchmark_symbol].iloc[0])
        sp500_cumulative = sp500_cumulative.dropna()
        sp500_cumulative.index = pd.to_datetime(sp500_cumulative.index)  # Ensure proper datetime index
        print("S&P 500 cumulative head:")
        print(sp500_cumulative.head())
    except Exception as e:
        print("Error in S&P 500 calculation:", e)
        sp500_cumulative = pd.Series(dtype=float)
else:
    print("S&P 500 data not available!")
    sp500_cumulative = pd.Series(dtype=float)

# Calculate total rate of return
portfolio_return = (portfolio_cumulative.iloc[-1] - 1) * 100 if not portfolio_cumulative.empty else None
sp500_return = (sp500_cumulative.iloc[-1] - 1) * 100 if not sp500_cumulative.empty else None

# Plot cumulative returns over 20 years
plt.figure(figsize=(12, 6))
if not portfolio_cumulative.empty:
    plt.plot(portfolio_cumulative.index, portfolio_cumulative, label=f'Portfolio (Return: {portfolio_return:.2f}%)', linewidth=2, color='blue')
    print("Plotting portfolio cumulative return.")  # Debugging output
else:
    print("Error: portfolio_cumulative is empty, no plot available.")

if not sp500_cumulative.empty:
    plt.plot(sp500_cumulative.index, sp500_cumulative, label=f'S&P 500 (Return: {sp500_return:.2f}%)', linestyle='dashed', color='red')
    print("Plotting S&P 500 cumulative return.")  # Debugging output
else:
    print("Error: S&P 500 cumulative return is empty, no plot available.")

plt.title("Cumulative Growth of Portfolio Over 20 Years")
plt.xlabel("Year")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.show()

"""Potforlio Returns Range Year 1 - 10"""

# Calculate log returns (excluding S&P 500 for portfolio calculations)
log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
mean_returns = log_returns[valid_stocks].mean().to_numpy()  # Only use portfolio stocks
std_dev = log_returns[valid_stocks].std().to_numpy()  # Only use portfolio stocks

# Define portfolio weights (equal weighting for now)
num_stocks = len(valid_stocks)
weights = np.array([1/num_stocks] * num_stocks) if num_stocks > 0 else np.array([])
print("Weights:", weights)  # Debugging output

# Monte Carlo Simulation for portfolio returns over different years
years = [1, 3, 5, 7, 10]
percentile_5th, median, percentile_95th = [], [], []

for year in years:
    days = year * 252
    simulated_returns = []

    for _ in range(simulation_runs):
        daily_drift = (mean_returns - 0.5 * std_dev ** 2) / 252  # Scale drift per day
        daily_shock = std_dev / np.sqrt(252)  # Scale volatility per day

        # Generate daily returns using GBM
        random_shocks = np.random.normal(0, 1, (num_stocks, days))
        daily_returns = np.exp(daily_drift[:, np.newaxis] + daily_shock[:, np.newaxis] * random_shocks)
        price_paths = np.cumprod(daily_returns, axis=1)  # Cumulative product to get price paths
        simulated_portfolio_return = np.dot(weights, price_paths[:, -1]) - 1  # Final portfolio return
        simulated_returns.append(simulated_portfolio_return)

    percentile_5th.append(np.percentile(simulated_returns, 5) * 100)
    median.append(np.percentile(simulated_returns, 50) * 100)
    percentile_95th.append(np.percentile(simulated_returns, 95) * 100)

# Define years for simulation
years = [1, 3, 5, 7, 10]

# Generate random sample data for illustration (replace with actual simulation results)
np.random.seed(42)  # For reproducibility
simulation_runs = 1000
simulated_returns = np.random.normal(loc=0.07, scale=0.15, size=(len(years), simulation_runs)) * 100  # Convert to percentage

# Compute actual percentiles from simulated data
percentile_5th = np.percentile(simulated_returns, 5, axis=1)
median = np.percentile(simulated_returns, 50, axis=1)
percentile_95th = np.percentile(simulated_returns, 95, axis=1)

# Create a DataFrame with years as rows
percent_return_df = pd.DataFrame({
    "5th Percentile (%)": percentile_5th,
    "Median (%)": median,
    "95th Percentile (%)": percentile_95th
}, index=years)

# Rename index to "Years"
percent_return_df.index.name = "Years"

# Print results
print(percent_return_df)

# Create a stacked bar chart ensuring 5th and 95th percentiles have the exact same color
fig, ax = plt.subplots(figsize=(8, 6))

# Define common color for 5th and 95th percentile ranges
percentile_color = '#1f77b4'  # Standard blue color

# Plot stacked bars with the exact same color for 5th and 95th percentiles
ax.bar(years, median - percentile_5th, bottom=percentile_5th, color='gray', label='Median')
ax.bar(years, percentile_95th - median, bottom=median, color=percentile_color, label='95th Percentile')
ax.bar(years, percentile_5th, color=percentile_color, label='5th Percentile')

# Labels and title
ax.set_xticks(years)
ax.set_xticklabels([f"Year {y}" for y in years])
ax.set_ylabel("Return (Percent)")
ax.set_title("Monte Carlo Simulation: Percentile Return Ranges")

# Increase Y-axis limits for better visibility
ax.set_ylim(min(percentile_5th) - 30, max(percentile_95th) + 30)

# Legend
ax.legend(loc="upper left")

# Show the plot
plt.show()

# Histogram of the portfolio returns using the simulated data
fig, ax = plt.subplots(figsize=(8, 6))

# Flatten the simulated returns data for histogram plotting
all_simulated_returns = simulated_returns.flatten()

# Plot histogram
ax.hist(all_simulated_returns, bins=50, color='blue', alpha=0.7, edgecolor='black')

# Labels and title
ax.set_xlabel("Portfolio Return (%)")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of Portfolio Returns (Monte Carlo Simulation)")

# Show the plot
plt.show()

initial_portfolio_value = 1000000

# Monte Carlo Simulation based on historical stock statistics
simulation_runs = 1000  # Number of Monte Carlo simulations
years_projection = np.arange(0, 21)  # Years 0 to 20
time_horizon = len(years_projection) * 252  # Convert years to trading days

# Simulated daily log returns for each stock
historical_stock_returns = np.random.normal(loc=0.0005, scale=0.01, size=(len(dates), num_stocks))

# Convert to DataFrame with stock symbols as columns
historical_stock_returns_df = pd.DataFrame(historical_stock_returns, index=dates, columns=stock_symbols)

# Compute portfolio log returns using equal weighting
portfolio_log_returns = historical_stock_returns_df.mean(axis=1)

# Calculate historical mean return and volatility for the portfolio
historical_mean_return = portfolio_log_returns.mean() * 252  # Annualized mean return
historical_volatility = portfolio_log_returns.std() * np.sqrt(252)  # Annualized volatility

# Simulate portfolio returns using historical mean return and volatility
simulated_portfolio_returns = np.exp(
    np.cumsum(np.random.normal(historical_mean_return / 252, historical_volatility / np.sqrt(252),
                               (simulation_runs, time_horizon)), axis=1)
)

# Scale initial portfolio value
simulated_portfolio_values = simulated_portfolio_returns * initial_portfolio_value

# Extract values at each year marker
portfolio_projections = {p: [] for p in [10, 25, 50, 75, 90]}
for year in years_projection:
    day_index = year * 252 if year * 252 < time_horizon else -1  # Avoid out-of-bounds indexing
    for p in portfolio_projections.keys():
        portfolio_projections[p].append(np.percentile(simulated_portfolio_values[:, day_index], p))

# Create a DataFrame to display portfolio values in dollars for different percentiles
portfolio_value_df = pd.DataFrame(portfolio_projections, index=years_projection)

# Rename columns for better readability
portfolio_value_df.columns = [f"{p}th Percentile ($)" for p in portfolio_projections.keys()]

# Rename index to "Years"
portfolio_value_df.index.name = "Years"

# Round values to the nearest dollar
portfolio_value_df = portfolio_value_df.round(0).astype(int)

# Print results
print(portfolio_value_df)

import matplotlib.ticker as ticker

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each percentile projection
for p in portfolio_projections.keys():
    ax.plot(years_projection, portfolio_value_df[f"{p}th Percentile ($)"], label=f"{p}th Percentile")

# Labels and title
ax.set_xlabel("Years")
ax.set_ylabel("Portfolio Value ($)")
ax.set_title("Projected Market Portfolio Value Over 20 Years Based on Selected Stocks")
ax.set_xticks(years_projection)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Format y-axis as currency
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Show the plot
plt.show()

# Compute percentage returns relative to the initial portfolio value ($1M)
portfolio_return_df = (portfolio_value_df / initial_portfolio_value - 1) * 100

# Round to two decimal places for readability
portfolio_return_df = portfolio_return_df.round(2)

# Print results
print(portfolio_return_df)

"""Momentum Strategy"""

# Define momentum-based Monte Carlo simulation parameters
lookback_period = 252  # 1-year lookback for momentum signal
momentum_threshold = 0.02  # Momentum threshold for overweighting trending stocks

# momentum scores (rolling 1-year return for each stock)
momentum_scores = historical_stock_returns_df.rolling(lookback_period).mean()

# Assign portfolio weights based on momentum (higher weight for higher momentum stocks)
momentum_weights = momentum_scores.div(momentum_scores.abs().sum(axis=1), axis=0).fillna(1/num_stocks)

# momentum-weighted portfolio returns
momentum_portfolio_returns = (historical_stock_returns_df * momentum_weights.shift(1)).sum(axis=1)

# historical mean return and volatility for the momentum strategy
momentum_mean_return = momentum_portfolio_returns.mean() * 252  # Annualized return
momentum_volatility = momentum_portfolio_returns.std() * np.sqrt(252)  # Annualized volatility

# Monte Carlo Simulation based on momentum-adjusted returns
simulated_momentum_returns = np.exp(
    np.cumsum(np.random.normal(momentum_mean_return / 252, momentum_volatility / np.sqrt(252),
                               (simulation_runs, time_horizon)), axis=1)
)

# Scale initial portfolio value
simulated_momentum_values = simulated_momentum_returns * initial_portfolio_value

# Extract values at each year marker
momentum_portfolio_projections = {p: [] for p in [10, 25, 50, 75, 90]}
for year in years_projection:
    day_index = year * 252 if year * 252 < time_horizon else -1  # Avoid out-of-bounds indexing
    for p in momentum_portfolio_projections.keys():
        momentum_portfolio_projections[p].append(np.percentile(simulated_momentum_values[:, day_index], p))

# Create a DataFrame to display momentum strategy portfolio values
momentum_portfolio_value_df = pd.DataFrame(momentum_portfolio_projections, index=years_projection)

# Rename columns for better readability
momentum_portfolio_value_df.columns = [f"{p}th Percentile ($)" for p in momentum_portfolio_projections.keys()]

# Rename index to "Years"
momentum_portfolio_value_df.index.name = "Years"

# Round values to the nearest dollar
momentum_portfolio_value_df = momentum_portfolio_value_df.round(0).astype(int)

# Display results
print(momentum_portfolio_value_df)

# Re-plot the Monte Carlo simulation results for momentum strategy
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each percentile projection
for p in momentum_portfolio_projections.keys():
    ax.plot(years_projection, momentum_portfolio_value_df[f"{p}th Percentile ($)"], label=f"{p}th Percentile")

# Labels and title
ax.set_xlabel("Years")
ax.set_ylabel("Portfolio Value ($)")
ax.set_title("Monte Carlo Simulation with Momentum Strategy Over 20 Years")
ax.set_xticks(years_projection)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Format y-axis as currency
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Show the plot
plt.show()