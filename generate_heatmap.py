import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bcs import minkowski_distance
from strategy import Strategy
from trading_context import TradingContext


def excess_sharpe_df(p_range, tol_range, target_weights, tc):
    """Creates a dataframe of the excess Sharpe for each rebalancing strategy over the buy-and-hold portfolio.

    :param p_range: A tuple containing the minimum, maximum and step for the Minkowski p-value
    :type p_range: tuple
    :param tol_range: A tuple containing the minimum, maximum and step for the rebalancing tolerance
    :type tol_range: tuple
    :param target_weights: A series containing the target weighting of the instruments. Also the initial weights
    :type target_weights: pandas.Series
    :param tc: A trading context containing information regarding the traded instruments as well as initial cash and \
        commission
    :type tc: trading_context.TradingContext
    :return: a dataframe containing the excess Sharpe ratio of each rebalanced strategy in the p_range and tol_range
    """
    min_p, max_p, step_p = p_range
    min_tol, max_tol, step_tol = tol_range
    sharpe_df = pd.DataFrame(
        columns=np.round(np.arange(min_tol, max_tol, step_tol), 2),
        index=np.arange(min_p, max_p, step_p),
        dtype=np.float64
    )

    # Since our baseline strategy is the buy-and-hold portfolio, we calculate the excess Sharpe, i.e. the increase
    # in the Sharpe ratio of switching from the buy-and-hold portfolio to the rebalanced portfolio
    buy_and_hold = Strategy('buy_and_hold', target_weights, tc)
    buy_and_hold_sharpe = buy_and_hold.summary_stats()['Sharpe Ratio']

    # TQDM is a progress bar library, useful for keeping track of how many iterations are left.
    for p in tqdm(sharpe_df.index, desc='p values'):
        for tol in tqdm(sharpe_df.columns, desc='tolerances'):
            # We create a new strategy for each p and tol, and calculate the excess Sharpe
            rebalanced = Strategy('Rebalanced', target_weights, tc)
            rebalanced.trade(trigger_function=minkowski_distance, trigger_point=tol, p=p)
            stats = rebalanced.summary_stats()
            sharpe_df.loc[p, tol] = stats['Sharpe Ratio'] - buy_and_hold_sharpe

    sharpe_df.columns.name = 'Threshold'
    sharpe_df.index.name = 'Minkowski p'

    sharpe_df = sharpe_df.round(3)

    return sharpe_df


def run(starting_cash=100_000, commission=0.005):
    """Create a range of Strategies, each with different parameters for the Minkowski p-value and rebalancing
    tolerance. The Sharpe ratio of each portfolio is then subtracted by the Sharpe ratio of the buy-and-hold
    portfolio to find the excess Sharpe ratio. These ratios are then saved to `heatmap.csv` and a heatmap of the ratios
    are produced to `heatmap.png`.

    :param starting_cash: The amount of cash to which the strategy has access
    :type starting_cash: float
    :param commission: The proportion of total value paid in fees during a rebalance
    :type commission: float
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
    p_range = 1, 10, 1
    tol_range = 0.01, 0.2, 0.01
    sharpe_df = excess_sharpe_df(p_range, tol_range, target_weights, tc)

    sharpe_df.to_csv(os.path.join('datasets', 'heatmap.csv'))

    # A mask is created to cover any cells where the excess Sharpe is 0, i.e., the strategy did not rebalance
    mask = sharpe_df == 0

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sharpe_df, linewidths=0.1, ax=ax, annot=True, fmt='.3g', cmap="gray_r", mask=mask)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'))
    plt.gcf().clear()
    plt.close()


if __name__ == '__main__':
    run(starting_cash=100_000, commission=0.005)
