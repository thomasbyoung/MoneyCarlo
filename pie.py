import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DATA_DIR = os.path.join(os.getcwd(), "data")  
sp500_csv = os.path.join(DATA_DIR, "SPY_HistoricalData.csv")  

def load_etf_data(data_dir):
    etf_data = pd.DataFrame()
    etf_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "SPY" not in f]
    stock_list = [f.split("_")[0] for f in etf_files]  

    for file in etf_files:
        ticker = file.split("_")[0]
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        if "Close/Last" in df.columns:
            df = df.rename(columns={"Close/Last": "Close"})

        if "Close" not in df.columns:
            raise ValueError(f"Missing 'Close' column in {file}")

        df = df.rename(columns={"Close": ticker})

        if etf_data.empty:
            etf_data = df[[ticker]]
        else:
            etf_data = etf_data.join(df[[ticker]], how="outer")

    etf_data.dropna(inplace=True) 
    return etf_data, stock_list

def monte_carlo_simulation(meanReturns, covMatrix, weights, T, mc_sims, initialPortfolio):
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T 

    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

    return portfolio_sims

def plot_simulation(portfolio_sims):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('Monte Carlo Simulation')
    plt.show()

def calculate_failure_rate(portfolio_sims, initialPortfolio, expected_gain):
    T = portfolio_sims.shape[0]
    Loss = portfolio_sims[T - 1] < initialPortfolio * expected_gain
    nb_losses = np.sum(Loss)
    failure_rate = (nb_losses / portfolio_sims.shape[1]) * 100
    return failure_rate

def historical_comparison(etf_data, sp500_csv, weights):
    sp500_data = pd.read_csv(sp500_csv, parse_dates=['Date'], index_col='Date')

    if "Close/Last" in sp500_data.columns:
        sp500_data = sp500_data.rename(columns={"Close/Last": "Close"})

    if "Close" not in sp500_data.columns:
        raise ValueError("SPY_HistoricalData.csv must have a 'Close' column.")

    sp500_data.sort_index(inplace=True)
    sp500_returns = sp500_data['Close'].pct_change().cumsum()

    portfolio_returns = etf_data.pct_change()
    weighted_portfolio_returns = portfolio_returns * weights
    daily_weighted_return = weighted_portfolio_returns.sum(axis=1).cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_weighted_return.index, daily_weighted_return, label='ETF Portfolio Return')
    plt.plot(sp500_returns.index, sp500_returns, label='S&P 500 (SPY) Return')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('ETF Portfolio vs. S&P 500')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def save_results_to_file(output_text, filename="simulation_results.txt"):
    with open(filename, "w") as f:
        f.write(output_text)
    print(output_text) 


if __name__ == '__main__':
    etf_data, stock_list = load_etf_data(DATA_DIR)
    returns = etf_data.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    weights = np.random.random(len(stock_list))
    weights /= np.sum(weights)
    mc_sims = 10000  
    T = 365  
    initialPortfolio = 100000  
    portfolio_sims = monte_carlo_simulation(meanReturns, covMatrix, weights, T, mc_sims, initialPortfolio)

    expected_gain = 1.1  
    failure_rate = calculate_failure_rate(portfolio_sims, initialPortfolio, expected_gain)
    
    final_values = portfolio_sims[-1, :]
    avg_final_value = np.mean(final_values)
    std_dev_final_value = np.std(final_values)
    percentiles = np.percentile(final_values, [10, 25, 50, 75, 90])
    best_case = np.max(final_values)
    worst_case = np.min(final_values)

    portfolio_annual_return = returns.mean().mean() * 252
    portfolio_annual_volatility = returns.std().mean() * np.sqrt(252)
    sharpe_ratio = portfolio_annual_return / portfolio_annual_volatility
    max_drawdown = (etf_data / etf_data.cummax() - 1).min().min()

    results_text = f"""
    === Monte Carlo Simulation Results ===
    Initial Portfolio Value: ${initialPortfolio:,.2f}
    Expected Gain (Target): {expected_gain*100:.1f}%
    Failure Rate: {failure_rate:.2f}%
    
    Portfolio Value Distribution:
    - 10th Percentile: ${percentiles[0]:,.2f}
    - 25th Percentile: ${percentiles[1]:,.2f}
    - 50th Percentile (Median): ${percentiles[2]:,.2f}
    - 75th Percentile: ${percentiles[3]:,.2f}
    - 90th Percentile: ${percentiles[4]:,.2f}
    
    - Average Final Portfolio Value: ${avg_final_value:,.2f}
    - Standard Deviation: ${std_dev_final_value:,.2f}
    - Best Case: ${best_case:,.2f}
    - Worst Case: ${worst_case:,.2f}

    === ETF Portfolio vs. S&P 500 ===
    Portfolio Annualized Return: {portfolio_annual_return:.2%}
    Portfolio Annualized Volatility: {portfolio_annual_volatility:.2%}
    Sharpe Ratio: {sharpe_ratio:.2f}
    Maximum Drawdown: {max_drawdown:.2%}

    === ETF Weights Used in Simulation ===
    {dict(zip(stock_list, weights))}

    === ETF Correlation Matrix ===
    {covMatrix}
    """

    save_results_to_file(results_text)

    plot_simulation(portfolio_sims)
    historical_comparison(etf_data, sp500_csv, weights)
