import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bcs import Strategy


def generate_heatmap(returns_df, target_weights, starting_cash, commission):
    dates = returns_df.index
    tickers = returns_df['daily'].columns
    min_p, max_p, step_p = 1, 10, 1
    min_tol, max_tol, step_tol = 0.01, 0.2, 0.01
    sharpe_df = pd.DataFrame(
        columns=np.round(np.arange(min_tol, max_tol, step_tol), 2),
        index=np.arange(min_p, max_p, step_p),
        dtype=np.float64
    )
    for p in tqdm(sharpe_df.index, desc='p values'):
        for tol in tqdm(sharpe_df.columns, desc='tolerances'):
            rebalanced = Strategy('Rebalanced', dates, tickers, returns_df, target_weights, starting_cash, commission)
            rebalanced.trade(p, tol)
            stats = rebalanced.summary_stats()
            sharpe_df.loc[p, tol] = stats[4]

    sharpe_df.columns.name = 'Threshold'
    sharpe_df.index.name = 'Minkowski p'

    buy_and_hold = Strategy('buy_and_hold', dates, tickers, returns_df, target_weights, starting_cash, commission)
    buy_and_hold_sharpe = buy_and_hold.summary_stats()[4]
    sharpe_df -= buy_and_hold_sharpe
    mask = sharpe_df == 0
    sharpe_df = sharpe_df.round(3)

    sharpe_df.to_csv(os.path.join('datasets', 'sharpe.csv'))



    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sharpe_df, linewidths=0.1, ax=ax, annot=True, fmt='.3g', cmap="gray_r", xticklabels=2, yticklabels=2,
                mask=mask)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'))
    plt.gcf().clear()
    plt.close()


def run(starting_cash=100_000, commission=0.005):
    sns.set(style='whitegrid')

    returns_df = pd.read_csv(
        filepath_or_buffer='portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    returns_df.index.name = 'Date'
    returns_df.columns = pd.MultiIndex.from_product([['daily'], tickers])
    target_weights = pd.Series(data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05], index=tickers)
    returns_df[list(zip(['cumulative'] * 8, tickers))] = (returns_df['daily'] + 1).cumprod()

    generate_heatmap(returns_df, target_weights, starting_cash, commission)


if __name__ == '__main__':
    run(starting_cash=100_000, commission=0.005)

