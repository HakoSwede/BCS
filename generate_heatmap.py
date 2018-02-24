import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bcs import minkowski_distance
from strategy import Strategy
from trading_context import TradingContext


def generate_heatmap(p_range, tol_range, target_weights, tc):
    """Creates a heatmap of the excess Sharpe for each rebalancing stategy over the buy-and-hold portfolio.

    :param p_range:
    :param tol_range:
    :param target_weights:
    :param tc:
    :return:
    """
    min_p, max_p, step_p = p_range
    min_tol, max_tol, step_tol = tol_range
    sharpe_df = pd.DataFrame(
        columns=np.round(np.arange(min_tol, max_tol, step_tol), 2),
        index=np.arange(min_p, max_p, step_p),
        dtype=np.float64
    )

    # TQDM is a progress bar library, useful for keeping track of how many iterations are left.
    for p in tqdm(sharpe_df.index, desc='p values'):
        for tol in tqdm(sharpe_df.columns, desc='tolerances'):
            rebalanced = Strategy('Rebalanced', target_weights, tc)
            rebalanced.trade(trigger_function=minkowski_distance, trigger_point=tol, p=p)
            stats = rebalanced.summary_stats()
            sharpe_df.loc[p, tol] = stats['Sharpe Ratio']

    sharpe_df.columns.name = 'Threshold'
    sharpe_df.index.name = 'Minkowski p'

    # Since our baseline strategy is the buy-and-hold portfolio, we calculate the excess Sharpe, i.e. the increase
    # in the Sharpe ratio of switching from the buy-and-hold portfolio to the rebalanced portfolio
    buy_and_hold = Strategy('buy_and_hold', target_weights, tc)
    buy_and_hold_sharpe = buy_and_hold.summary_stats()['Sharpe Ratio']
    sharpe_df -= buy_and_hold_sharpe
    mask = sharpe_df == 0
    sharpe_df = sharpe_df.round(3)

    sharpe_df.to_csv(os.path.join('datasets', 'heatmap.csv'))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sharpe_df, linewidths=0.1, ax=ax, annot=True, fmt='.3g', cmap="gray_r", mask=mask)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'))
    plt.gcf().clear()
    plt.close()


def run(p_range=(1, 10, 1), tol_range=(0.01, 0.2, 0.01), starting_cash=100_000, commission=0.005):
    """

    :param p_range:
    :param tol_range:
    :param starting_cash:
    :param commission:
    :return:
    """
    sns.set(style='whitegrid')
    returns_df = pd.read_csv(
        filepath_or_buffer='portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )
    returns_df.index.name = 'Date'

    tc = TradingContext(
        dates=returns_df.index,
        tickers=returns_df.columns,
        instrument_returns=returns_df,
        starting_cash=starting_cash,
        commission=commission
    )

    target_weights = pd.Series(data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05], index=tc.tickers)
    generate_heatmap(p_range, tol_range, target_weights, tc)


if __name__ == '__main__':
    p_range = 1, 10, 1
    tol_range = 0.01, 0.2, 0.01
    run(p_range, tol_range, starting_cash=100_000, commission=0.005)
