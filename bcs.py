import os
from functools import partial

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pandas as pd
import seaborn as sns

# Declaration of constants
BETTERMENT_BLUE = '#1F4AB4'
BETTERMENT_GRAY = '#30363D'
BETTERMENT_PALETTE = [
    '#79CCFF',  # SHY
    '#ADE7FF',  # TIP
    '#B5FFCB',  # VTI
    '#41F39A',  # IVE
    '#30EC9B',  # IWN
    '#19DB9A',  # IWS
    '#218080',  # EFA
    '#112F42',  # EEM
]


class Strategy:
    def __init__(self, name, dates, tickers, etf_returns, target_weights, starting_cash, commission):
        self.name = name
        self.dates = dates
        self.tickers = tickers
        self.target_weights = target_weights
        self.starting_cash = starting_cash
        self.commission = commission
        columns = pd.MultiIndex.from_product([['values', 'allocations'], tickers])
        self.df = pd.DataFrame(data=np.zeros((len(dates),len(columns))),index=dates, columns=columns, dtype=np.float64)
        self.etf_returns = etf_returns
        self.initialize_df()
        self.trades = pd.DataFrame(columns=self.tickers, dtype=np.float64)
        self.trades.loc[self.dates[0]] = self.df.loc[self.dates[0], 'values'].values  # First trade is purchasing target portfolio.

    def initialize_df(self):
        self.df['values'] = (self.etf_returns['cumulative'] * self.starting_cash).mul(self.target_weights, axis=1).values
        self.df['allocations'] = (self.df['values'].div(self.df['values'].sum(axis=1), axis=0))
        self.df['returns'] = (self.df['values'].sum(axis=1)).pct_change(1).fillna(0)


    def summary_stats(self):
        total_return = self.df['values'].iloc[-1].sum() / self.starting_cash
        seconds_invested = (self.df.index[-1] - self.df.index[0]).total_seconds()
        seconds_per_year = 60 * 60 * 24 * 365
        annualized_returns = total_return ** (seconds_per_year / seconds_invested) - 1
        annualized_volatility = self.df['returns'].std() * (252 ** 0.5)
        sharpe = annualized_returns / annualized_volatility
        return annualized_returns, annualized_volatility, sharpe

    def rebalance(self, date):
        # If we are not within tolerance, we rebalance. Rebalancing is done at the end of the trading day,
        # which is why we still grow the portfolio by the daily returns.
        eod_values = self.df.shift(1).loc[date, 'values'].mul(1 + self.etf_returns.loc[date, 'daily'])
        eod_portfolio_value = sum(eod_values.values)

        previous_values = self.df.loc[date, 'values'].copy()
        position_value = eod_portfolio_value * self.target_weights
        trading_cost = abs(eod_values.div(eod_portfolio_value) - self.target_weights) * eod_portfolio_value * self.commission
        current_values = position_value - trading_cost
        self.df.loc[date, 'values'] = current_values.values
        future_values = self.etf_returns.loc[date:, 'cumulative'].div(
            self.etf_returns.loc[date, 'cumulative']).mul(current_values, axis=1)
        self.df.loc[date:, 'values'] = future_values.values
        trade = pd.Series(current_values - previous_values)
        # Once we have calculated the end-of-day value of the portfolio, we set the allocation by looking at the
        # dollars invested in each ETF
        self.df.loc[date:, 'allocations'] = future_values.div(future_values.sum(axis=1), axis=0).values

        return trade

    def trade(self, minkowski_p, max_drift):
        """
            Main method to implement the specified trading strategy. The strategy will rebalance whenever the max_drift is less
            than the allowed drift based on the Minkowski p-value and the specified target weights.
            :param strategy: The strategy to trade
            :param minkowski_p: The p-value for the Minkowski distance measure
            :param max_drift: The max allowed percentage point drift from the strategies ideal weighting
            :return: None
            """
        current_drift = partial(minkowski_distance, arr_2=self.target_weights, p=minkowski_p)
        for date in self.dates[1:]:
            # If the previous-day close allocation is out of tolerance..
            if current_drift(self.df.shift(1).loc[date, 'allocations'].values) > max_drift:
                # then rebalance the portfolio
                trade = self.rebalance(date)
                self.trades.loc[date] = trade
        self.df['returns'] = self.df['values'].sum(axis=1).pct_change(1).fillna(0)

    def save_to_csv(self):
        path = partial(os.path.join, 'datasets')
        self.df['values'].sum(axis=1).to_csv(path('{0}_values.csv'.format(self.name)))
        self.df['allocations'].to_csv(path('{0}_allocations.csv'.format(self.name)))
        self.df['returns'].to_csv(path('{0}_returns.csv'.format(self.name)))
        self.trades.to_csv(path('{0}_trades.csv'.format(self.name)))

def minkowski_distance(arr_1, arr_2, p):
    """
    An implementation of the metric for the Lebesgue spaces. The Minkowski distance generalizes to many
    well-known metrics for specific choices of p. For example:

    p = 1: Manhattan Distance

    p = 2: Euclidian distance

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
    General function for plotting images of the dataframes created by the main code of the file.

    :param strategy_!: The first strategy to plot and compare. This is the benchmark.
    :param strategy_2: The second strategy to plot and compare. This is the portfolio.
    :return:
    """

    # RETURNS PLOT
    df_1 = strategy_1.df
    df_2 = strategy_2.df
    df_trades = strategy_2.trades
    df_1_ax = plt.subplot2grid((2, 2), (0, 0))
    df_2_ax = plt.subplot2grid((2, 2), (1, 0))
    hist_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    df_1['returns'].plot(
        ax=df_1_ax,
        figsize=(12, 6),
        title='Daily returns of buy-and-hold portfolio',
        color=BETTERMENT_GRAY
    )
    df_2['returns'].plot(
        ax=df_2_ax,
        title='Daily returns of rebalanced portfolio',
        color=BETTERMENT_BLUE
    )
    sns.distplot(
        a=df_1['returns'],
        color=BETTERMENT_GRAY,
        ax=hist_ax,
        bins=50,
        label='Buy and Hold'
    )
    sns.distplot(
        a=df_2['returns'],
        color=BETTERMENT_BLUE,
        ax=hist_ax,
        bins=50,
        label='Rebalance'
    )
    plt.title('Frequency of returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'daily_returns.png'), dpi=300)
    plt.gcf().clear()

    # ALLOCATIONS PLOT
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')

    df_1['allocations'].plot(
        ax=axes_alloc[0],
        figsize=(12, 6),
        title='weight of portfolio assets of buy-and-hold portfolio',
        kind='area',
        legend=False,
        ylim=(0, 1),
        colormap=cm.get_cmap('betterment', 8)
    )
    df_2['allocations'].plot(
        ax=axes_alloc[1],
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        legend=False,
        colormap=cm.get_cmap('betterment')
    )
    axes_alloc[0].set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=8)
    plt.savefig(os.path.join('images', 'asset_allocations.png'), dpi=300)
    plt.gcf().clear()

    # PORTFOLIO VALUE PLOT
    fig_values, axes_values = plt.subplots()

    df_1['values'].sum(axis=1).plot(
        ax=axes_values,
        figsize=(12, 6),
        title='Portfolio values',
        label='Value of buy-and-hold portfolio',
        color=BETTERMENT_GRAY
    )
    df_2['values'].sum(axis=1).plot(
        ax=axes_values,
        label='Value of rebalanced portfolio',
        color=BETTERMENT_BLUE
    )
    axes_values.set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.savefig(os.path.join('images', 'values.png'), dpi=300)
    plt.gcf().clear()
    plt.close()

    # TRADES PLOT
    fig_trades, axes_trades = plt.subplots()

    df_trades.cumsum().reindex(df_1.index, method='ffill').plot(
        ax=axes_trades,
        figsize=(12, 6),
        title='Cumulative investment per ETF',
        cmap='betterment'
    )
    axes_trades.set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig(os.path.join('images', 'trades.png'), dpi=300)
    plt.gcf().clear()
    plt.close()


def run():
    max_drift = 0.05  # Maximum distance from optimal portfolio
    minkowski_p = 5  # Minkowski-p to determine which distance measure to use
    starting_cash = 100_000
    commission = 0.005  # This is a form of commission (i.e. fees paid per trade), expressed in percentage.
    cm.register_cmap('betterment', cmap=colors.ListedColormap(BETTERMENT_PALETTE))
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
    buy_and_hold = Strategy('buy_and_hold', dates, tickers, returns_df, target_weights, starting_cash, commission)

    # REBALANCED PORTFOLIO
    # The rebalanced portfolio is our 'active' portfolio for this case study. It rebalances its holdings whenever the
    # allocation drifts too far from the target.
    rebalanced = Strategy('rebalanced', dates, tickers, returns_df, target_weights, starting_cash, commission)
    rebalanced.trade(minkowski_p, max_drift)

    save_images(buy_and_hold, rebalanced)
    buy_and_hold.save_to_csv()
    rebalanced.save_to_csv()


if __name__ == '__main__':
    run()
