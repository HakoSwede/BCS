import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from betterment_colors import BETTERMENT_BLUE, BETTERMENT_GRAY, BETTERMENT_PALETTE
from strategy import Strategy
from trading_context import TradingContext


def minkowski_distance(arr_1, arr_2, p):
    """
    An implementation of the metric for the Lebesgue spaces. The Minkowski distance generalizes to many
    well-known metrics for specific choices of p. For example:

    p = 1: Manhattan Distance

    p = 2: Euclidian distance

    p -> 0: Hamming distance

    p -> infinity: Chebyshev distance

    For a given p, the function will return the distance between the two points at arr_1 and arr_2 in L^p space

    :param arr_1: The location of the first point
    :type arr_1: array-like
    :param arr_2: The location of the second point
    :type arr_2: array-like
    :param p: The parameter specifying which p-norm will be used
    :type p: float
    :return: The distance between arr_1 and arr_2 in L^p space
    """
    return sum(abs(arr_1 - arr_2) ** p) ** (1 / p)


def save_images(strategy_1, strategy_2):
    """
    General function for plotting charts of two trading strategies. Charts produce include a stacked area chart of
    the asset allocation, a line chart for the daily returns of each trading strategy including a histogram,
    a line chart showing the net flows of cash between the instruments of the strategy, and a line chart
    showing the total returns of the two strategies. The charts are saved in the 'images' directory.

    :param strategy_1: The first strategy to plot and compare. This is the benchmark.
    :type strategy_1: Strategy
    :param strategy_2: The second strategy to plot and compare. This is the portfolio.
    :type strategy_2: Strategy
    :return: None
    """

    # RETURNS PLOT
    df_1_ax = plt.subplot2grid((2, 2), (0, 0))
    df_2_ax = plt.subplot2grid((2, 2), (1, 0))
    hist_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    strategy_1.returns_chart(df_1_ax, BETTERMENT_GRAY)
    strategy_2.returns_chart(df_2_ax, BETTERMENT_BLUE)
    strategy_1.returns_distribution_chart(hist_ax, BETTERMENT_GRAY)
    strategy_2.returns_distribution_chart(hist_ax, BETTERMENT_BLUE)
    plt.title('Frequency of returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'daily_returns.png'), dpi=300)
    plt.gcf().clear()

    # ALLOCATIONS PLOT
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    strategy_1.asset_allocations_chart(axes_alloc[0], cm.get_cmap('betterment'))
    strategy_2.asset_allocations_chart(axes_alloc[1], cm.get_cmap('betterment'))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=8)
    plt.savefig(os.path.join('images', 'asset_allocations.png'), dpi=300)
    plt.gcf().clear()

    # PORTFOLIO VALUE PLOT
    fig_values, axes_values = plt.subplots()
    strategy_1.value_chart(axes_values, BETTERMENT_GRAY)
    strategy_2.value_chart(axes_values, BETTERMENT_BLUE)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.savefig(os.path.join('images', 'values.png'), dpi=300)
    plt.gcf().clear()

    # TRADES PLOT
    fig_trades, axes_trades = plt.subplots()
    strategy_2.trades_chart(axes_trades, cm.get_cmap('betterment'))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig(os.path.join('images', 'trades.png'), dpi=300)
    plt.gcf().clear()
    plt.close()


def run(max_drift=0.05, minkowski_p=4, starting_cash=100000, commission=0.005):
    """

    :param max_drift: Our max allowed drift from the target portfolio, in percentage points.
    :type max_drift: float
    :param minkowski_p: The p-parameter for the Minkowski distance function
    :type minkowski_p: float
    :param starting_cash: The initial cash for the backtest
    :type starting_cash: float
    :param commission: The commission per trade, expressed in percentage points
    :type commission: float
    :return: None
    """
    cm.register_cmap('betterment', cmap=colors.ListedColormap(BETTERMENT_PALETTE))
    sns.set(style='whitegrid')
    returns_df = pd.read_csv(
        filepath_or_buffer='portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )

    # A trading context is created to ensure that both strategies are traded under the same conditions
    tc = TradingContext(
        dates=returns_df.index,
        tickers=returns_df.columns,
        instrument_returns=returns_df,
        starting_cash=starting_cash,
        commission=commission
    )

    target_weights = pd.Series(
        data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05],
        index=tc.tickers
    )

    # The buy-and-hold portfolio serves as our baseline. As expected from the name, the buy-and-hold portfolio buys
    # the target ETF portfolio and then holds it for the period.
    buy_and_hold = Strategy('Buy and Hold', target_weights, tc)

    # The rebalanced portfolio is our 'active' portfolio for this case study. It rebalances its holdings whenever the
    # allocation drifts too far from the target.
    rebalanced = Strategy('Rebalanced', target_weights, tc)

    # We give the rebalanced portfolio the Minkoski distance as its trigger function, and the max_drift as its
    # trigger point. The trade method then generates the portfolio returns over the trading period.
    rebalanced.trade(trigger_function=minkowski_distance, trigger_point=max_drift, p=minkowski_p)

    buy_and_hold_stats = buy_and_hold.summary_stats()
    rebalanced_stats = rebalanced.summary_stats()
    stats = pd.DataFrame(
        data=[buy_and_hold_stats, rebalanced_stats],
        index=['Buy and Hold', 'Rebalanced']
    )
    stats['Capital Gains'] = stats['Capital Gains'].round(2)
    for col in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']:
        stats[col] = stats[col].round(3)
    stats.index.name = 'Strategy'
    stats.to_csv(os.path.join('datasets', 'stats.csv'))

    save_images(buy_and_hold, rebalanced)
    buy_and_hold.save_to_csv()
    rebalanced.save_to_csv()


if __name__ == '__main__':
    run(max_drift=0.05, minkowski_p=4, starting_cash=100_000, commission=0.005)
