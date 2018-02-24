import os
from functools import partial

import numpy as np
import pandas as pd


class Strategy:
    """
    Data structure for a trading strategy.

    The strategy is initialized to a buy-and-hold strategy on the underlying instruments, with the initial weighting
    given by target_weights. At any point in time, the strategy can be rebalanced to the initial weighting by calling
    the rebalance method.

    :param name: Name of the strategy.
    :type name: string
    :param dates: A datetime index of the dates for which the strategy will be backtested
    :type dates: pandas.DatetimeIndex
    :param tickers: A list of the ticker symbols of the traded instruments
    :type tickers: list
    :param instrument_returns: A dataframe containing the daily and cumulative returns of the traded instruments
    :type instrument_returns: pandas.DataFrame
    :param target_weights: A series containing the target weighting of the instruments. Also the initial weights
    :type target_weights: pandas.Series
    :param starting_cash: The amount of cash to which the strategy has access
    :type starting_cash: float
    :param commission: The proportion of total value paid in fees during a rebalance
    :type commission: float
    """

    def __init__(self, name, dates, tickers, instrument_returns, target_weights, starting_cash, commission):
        self.name = name
        self.dates = dates
        self.tickers = tickers
        self.target_weights = target_weights
        self.starting_cash = starting_cash
        self.commission = commission
        # The strategy stores its values, returns and allocations internally as a dataframe
        self.df = pd.DataFrame(
            index=dates,
            columns=pd.MultiIndex.from_product([['values', 'allocations'], tickers]),
            dtype=np.float64
        )
        self.instrument_returns = instrument_returns
        self._initialize_df(self.df)
        self.trades = pd.DataFrame(columns=self.tickers, dtype=np.float64)
        # The first trade corresponds to buying the target weight of each instrument
        self.trades.loc[self.dates[0]] = self.df.loc[self.dates[0], 'values'].values

    def _initialize_df(self, df):
        """
        Initialization of the dataframe of the strategy.  The dataframe initially has the same returns as the
        'buy-and-hold' portfolio, and is then updated after every trade.

        :return: None
        """
        df['values'] = (self.instrument_returns['cumulative'] *
                        self.starting_cash).mul(self.target_weights, axis=1).values
        df['allocations'] = self.df['values'].div(df['values'].sum(axis=1), axis=0)
        df['returns'] = (df['values'].sum(axis=1)).pct_change(1).fillna(0)

    def rebalance(self, date):
        """
        Rebalance the strategy to its initial weighting at a given point in time. The strategy will then hold the
        instruments until it is rebalanced again.
        :param date: The date at which the strategy is to be rebalanced.
        :return:
        """

        eod_values = self.df.shift(1).loc[date, 'values'].mul(1 + self.instrument_returns.loc[date, 'daily'])
        eod_portfolio_value = sum(eod_values.values)

        previous_values = self.df.loc[date, 'values'].copy()
        position_value = eod_portfolio_value * self.target_weights
        trading_cost = abs(eod_values.div(eod_portfolio_value) - self.target_weights) * eod_portfolio_value * \
                       self.commission
        current_values = position_value - trading_cost
        self.df.loc[date, 'values'] = current_values.values
        future_values = self.instrument_returns.loc[date:, 'cumulative'].div(
            self.instrument_returns.loc[date, 'cumulative']).mul(current_values, axis=1)
        self.df.loc[date:, 'values'] = future_values.values
        trade = pd.Series(current_values - previous_values)
        # Once we have calculated the end-of-day value of the portfolio, we set the allocation by looking at the
        # dollars invested in each ETF
        self.df.loc[date:, 'allocations'] = future_values.div(future_values.sum(axis=1), axis=0).values

        return trade

    def trade(self, minkowski_p, max_drift):
        """
        Main method to implement the specified trading strategy. The strategy will rebalance whenever the max_drift
        is less than the allowed drift based on the Minkowski p-value and the specified target weights.
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

    def summary_stats(self):
        """

        :return:
        """
        capital_gains = self.df['values'].iloc[-1].sum() - self.starting_cash
        total_return = capital_gains / self.starting_cash
        seconds_invested = (self.df.index[-1] - self.df.index[0]).total_seconds()
        seconds_per_year = 60 * 60 * 24 * 365
        annualized_returns = (total_return + 1) ** (seconds_per_year / seconds_invested) - 1
        annualized_volatility = self.df['returns'].std() * (252 ** 0.5)
        sharpe = annualized_returns / annualized_volatility
        num_trades = self.trades.shape[0]
        return capital_gains, total_return, annualized_returns, annualized_volatility, sharpe, num_trades

    def save_to_csv(self):
        """

        :return:
        """
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
