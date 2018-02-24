import os
from functools import partial

import numpy as np
import pandas as pd


class Strategy:
    """

    """

    def __init__(self, name, dates, tickers, etf_returns, target_weights, starting_cash, commission):
        self.name = name
        self.dates = dates
        self.tickers = tickers
        self.target_weights = target_weights
        self.starting_cash = starting_cash
        self.commission = commission
        self.df = pd.DataFrame(
            data=np.zeros((len(dates), len(tickers) * 2)),
            index=dates,
            columns=pd.MultiIndex.from_product([['values', 'allocations'], tickers]),
            dtype=np.float64
        )
        self.etf_returns = etf_returns
        self.initialize_df()
        self.trades = pd.DataFrame(columns=self.tickers, dtype=np.float64)
        self.trades.loc[self.dates[0]] = self.df.loc[self.dates[0], 'values'].values


    def initialize_df(self):
        """

        :return:
        """
        self.df['values'] = (self.etf_returns['cumulative'] * self.starting_cash).mul(self.target_weights, axis=1).values
        self.df['allocations'] = (self.df['values'].div(self.df['values'].sum(axis=1), axis=0))
        self.df['returns'] = (self.df['values'].sum(axis=1)).pct_change(1).fillna(0)


    def summary_stats(self):
        """

        :return:
        """
        total_return = self.df['values'].iloc[-1].sum() / self.starting_cash
        seconds_invested = (self.df.index[-1] - self.df.index[0]).total_seconds()
        seconds_per_year = 60 * 60 * 24 * 365
        annualized_returns = total_return ** (seconds_per_year / seconds_invested) - 1
        annualized_volatility = self.df['returns'].std() * (252 ** 0.5)
        sharpe = annualized_returns / annualized_volatility
        return annualized_returns, annualized_volatility, sharpe

    def rebalance(self, date):
        """

        :param date:
        :return:
        """

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
