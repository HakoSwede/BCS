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
    :param target_weights: A series containing the target weighting of the instruments. Also the initial weights
    :type target_weights: pandas.Series
    :param tc:
    :type tc: TradingContext
    """

    def __init__(self, name, target_weights, tc):
        self.name = name
        self.target_weights = target_weights
        self.tc = tc
        # The strategy stores its values, returns and allocations internally as a dataframe
        self.df = pd.DataFrame(
            index=self.tc.dates,
            columns=pd.MultiIndex.from_product([['values', 'allocations'], tc.tickers]),
            dtype=np.float64
        )
        self._initialize_df(self.df)
        self.trades = pd.DataFrame(columns=self.tc.tickers, dtype=np.float64)
        # The first trade corresponds to buying the target weight of each instrument
        self.trades.loc[self.tc.dates[0]] = self.df.loc[self.tc.dates[0], 'values'].values

    def _initialize_df(self, df):
        """
        Initialization of the dataframe of the strategy.  The dataframe initially has the same returns as the
        'buy-and-hold' portfolio, and is then updated after every trade.

        :return: None
        """
        df['values'] = (self.tc.instrument_returns['cumulative'] *
                        self.tc.starting_cash).mul(self.target_weights, axis=1).values
        df['allocations'] = self.df['values'].div(df['values'].sum(axis=1), axis=0)
        df['returns'] = (df['values'].sum(axis=1)).pct_change(1).fillna(0)

    def rebalance(self, date):
        """
        Rebalance the strategy to its initial weighting at a given point in time. The strategy will then hold the
        instruments until it is rebalanced again.
        :param date: The date at which the strategy is to be rebalanced.
        :return:
        """

        eod_values = self.df.shift(1).loc[date, 'values'].mul(1 + self.tc.instrument_returns.loc[date, 'daily'])
        eod_portfolio_value = sum(eod_values.values)

        previous_values = self.df.loc[date, 'values'].copy()
        position_value = eod_portfolio_value * self.target_weights
        trading_cost = abs(eod_values.div(eod_portfolio_value) - self.target_weights) * eod_portfolio_value * \
                       self.tc.commission
        current_values = position_value - trading_cost
        self.df.loc[date, 'values'] = current_values.values
        future_values = self.tc.instrument_returns.loc[date:, 'cumulative'].div(
            self.tc.instrument_returns.loc[date, 'cumulative']).mul(current_values, axis=1)
        self.df.loc[date:, 'values'] = future_values.values
        trade = pd.Series(current_values - previous_values)
        # Once we have calculated the end-of-day value of the portfolio, we set the allocation by looking at the
        # dollars invested in each ETF
        self.df.loc[date:, 'allocations'] = future_values.div(future_values.sum(axis=1), axis=0).values

        return trade

    def trade(self, trigger_function, trigger_point, **trigger_function_kwargs):
        """
        Main function to implement trading strategy. The trigger_function has to accept two arrays, the current
        allocation and the target allocation. It can also accept additional arguments, given in **kwargs. The trigger
        function has to output a float, which can then be compared to the trigger_point. If the trigger_point is
        exceeded, the portfolio will rebalance.

        :param trigger_function: A trigger function that accepts two arrays and additional kwargs, and outputs a float
        :param trigger_point: A float determining at what point the portfolio is to be rebalanced
        :param trigger_function_kwargs: Additional keyword arguments for the trigger function
        :return: None
        """
        for date in self.tc.dates[1:]:
            # If the previous-day close allocation is out of tolerance..
            if trigger_function(self.df.shift(1).loc[date, 'allocations'].values, self.target_weights,
                                **trigger_function_kwargs) > trigger_point:
                # then rebalance the portfolio
                trade = self.rebalance(date)
                self.trades.loc[date] = trade
        self.df['returns'] = self.df['values'].sum(axis=1).pct_change(1).fillna(0)

    def summary_stats(self):
        """
        Return a series containing the summary statistics for a strategy

        :return: A pandas series containing capital gains, total return, annualized return, annualized volatility, \
            sharpe ratio, and number of trades.
        """
        capital_gains = self.df['values'].iloc[-1].sum() - self.tc.starting_cash
        total_return = capital_gains / self.tc.starting_cash
        days_invested = (self.df.index[-1] - self.df.index[0]).days
        annualized_returns = (total_return + 1) ** (365 / days_invested) - 1
        annualized_volatility = self.df['returns'].std() * (252 ** 0.5)
        sharpe = annualized_returns / annualized_volatility
        num_trades = self.trades.shape[0]
        stats = pd.Series(
            data=[capital_gains, total_return, annualized_returns, annualized_volatility, sharpe, num_trades],
            index=['Capital Gains', 'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
                   'Number of Trades']
        )
        return stats

    def save_to_csv(self):
        """
        Save the strategy to 4 separate CSVs. The CSVs contain data on the strategy values, allocations, returns and
        trades. The CSVs are saved in the datasets folder.

        :return: None
        """
        path = partial(os.path.join, 'datasets')
        self.df['values'].sum(axis=1).to_csv(path('{0}_values.csv'.format(self.name)))
        self.df['allocations'].to_csv(path('{0}_allocations.csv'.format(self.name)))
        self.df['returns'].to_csv(path('{0}_returns.csv'.format(self.name)))
        self.trades.to_csv(path('{0}_trades.csv'.format(self.name)))


