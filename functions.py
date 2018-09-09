# Beginning Setup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data as pdr

import fix_yahoo_finance as yf

yf.pdr_override()

pd.set_option('display.max_rows', 500)

# Function Definitons

def run(stock_code: str = "CPB", market_indices: [str] = ["^GSPC"], frequency: str = "D", start_date: str = "1980-01-01"):

    # Get Parsed Data
    stock_data = get_parsed_data(stock_code, frequency, start_date)
    market_data = [get_parsed_data(market_index, frequency, start_date) for market_index in market_indices]
    combined_data = merge_dfs(stock_code, stock_data, market_indices, market_data)
    generate_plot(combined_data)
    print(combined_data.isnull().values.any())
    return combined_data


def get_parsed_data(code: str, frequency: str, start_date: str) -> pd.DataFrame:

    # Raw Data from Yahoo
    raw_data = pdr.get_data_yahoo(code, start=start_date)["Adj Close"]
    raw_data.index.name = None

    # Return Formatted Dataframes
    return format_raw_prices(raw_data, frequency)


def format_raw_prices(raw_data: pd.Series, frequency: str) -> pd.DataFrame:

    # Resample to correct frequency
    df = pd.DataFrame({'date': raw_data.index, 'price': raw_data.values}).resample(frequency, on="date").last().reset_index(drop=True).dropna()
    # Calculate Daily Return
    prev_period_price = df["price"].shift(1)
    df["return"] = (df["price"] - prev_period_price) / prev_period_price

    return df

def merge_dfs(stock_code: str, stock_df: pd.DataFrame, market_indices: [str], market_dfs: [pd.DataFrame]) -> pd.DataFrame:

    stock_returns = stock_df.drop("price", axis=1)
    final_data = pd.DataFrame()
    for index, market_df in enumerate(market_dfs):
        market_return = stock_returns.merge(market_df.drop("price", axis=1), on="date", suffixes=["_stock", "_market"]).dropna(how="any")
        beta_value = stats.linregress(market_return["return_market"], market_return["return_stock"])
        print("Beta Value with Index" + market_indices[index] + ": " + str(beta_value.slope))
        market_return["market_index"] = market_indices[index]
        final_data = pd.concat([final_data, market_return])
    return final_data.dropna(how="any")

def generate_plot(data: pd.DataFrame):
    sns.set_style("white")

    # Plot the lines on two facets
    sns.lmplot(x="return_market", y="return_stock", data=data, height=10, aspect=1, hue="market_index", ci=None, truncate=True)
    # plt.xlim(-0.5, 1.5)
    # plt.ylim(-1,1)
    plt.show()