import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bcs import generate_rebalanced, calculate_summary_statistics, Strategy


def generate_sensitivity_plot(returns, df, target):
    df = df.copy()
    min_p, max_p, step_p = 1, 10, 1
    min_tol, max_tol, step_tol = 0.01, 0.2, 0.01
    sharpe_df = pd.DataFrame(
        columns=np.round(np.arange(min_tol, max_tol, step_tol), 2),
        index=np.arange(min_p, max_p, step_p),
        dtype=np.float64
    )
    for p in tqdm(sharpe_df.index, desc='p values'):
        for tol in tqdm(sharpe_df.columns, desc='tolerances'):
            rebalanced, _ = generate_rebalanced(returns, df, p, target, tol)
            _, _, sharpe = calculate_summary_statistics(rebalanced)
            sharpe_df.loc[p, tol] = sharpe

    sharpe_df.columns.name = 'Threshold'
    sharpe_df.index.name = 'Minkowski p'
    sharpe_df.to_csv(os.path.join('datasets', 'sharpe.csv'))

    min_sharpe = np.min(sharpe_df.min())
    mask = sharpe_df == min_sharpe

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sharpe_df, linewidths=0.1, ax=ax, annot=True, fmt='.3g', cmap="gray_r", xticklabels=2, yticklabels=2,
                mask=mask)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'))
    plt.gcf().clear()
    plt.close()


if __name__ == '__main__':
    max_drift = 0.05
    minkowski_p = 5
    sns.set(style='whitegrid')
    returns_df = pd.read_csv(
        filepath_or_buffer='portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    dates = returns_df.index
    returns_df.index.name = 'Date'
    returns_df.columns = pd.MultiIndex.from_product([['daily'], tickers])
    target_weights = pd.Series(data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05], index=tickers)

    returns_df[list(zip(['cumulative'] * 8, tickers))] = (returns_df['daily'] + 1).cumprod()

    # BUY AND HOLD PORTFOLIO
    # The buy-and-hold portfolio serves as our baseline. As expected from the name, the buy-and-hold portfolio buys
    # the target ETF portfolio and then holds it for the period.
    buy_and_hold = Strategy(dates, tickers, returns_df, target_weights)

    generate_sensitivity_plot(returns_df, buy_and_hold_df, target_weights)
