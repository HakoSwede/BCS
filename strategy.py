import os
from functools import partial

import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import trading_context


class Strategy:
    """Data structure for a trading strategy.

    The strategy is initialized to a buy-and-hold strategy on the underlying instruments, with the initial weighting
    given by target_weights. At any point in time, the strategy can be rebalanced to the initial weighting by calling
    the rebalance method.

    :param name: Name of the strategy.
    :type name: string
    :param target_weights: A series containing the target weighting of the instruments. Also the initial weights
    :type target_weights: pandas.Series
    :param tc: A trading context containing information regarding the traded instruments as well as initial cash and \
        commission
    :type tc: trading_context.TradingContext
    """

    def __init__(self, name, target_weights, tc):
        self.name = name
        self.target_weights = target_weights
        self.tc = tc
        # The strategy stores its values, returns and allocations internally as a dataframe
        self.df = pd.DataFrame(
            index=self.tc.dates,
            columns=pd.MultiIndex.from_product([['values', 'allocations'], self.tc.tickers]),
            dtype=np.float64
        )
        self._initialize_df(self.df)
        self.trades = pd.DataFrame(columns=self.tc.tickers, dtype=np.float64)
        # The first trade corresponds to buying the target weight of each instrument
        self.trades.loc[self.tc.dates[0]] = self.df.loc[self.tc.dates[0], 'values'].values

    def _initialize_df(self, df):
        """Initialization of the dataframe of the strategy.  The dataframe initially has the same returns as the
        'buy-and-hold' portfolio, and is then updated after every trade.

        :return: None
        """
        df['values'] = (self.tc.instrument_returns['cumulative'] *
                        self.tc.starting_cash).mul(self.target_weights, axis=1).values * (1 - self.tc.commission)
        df['allocations'] = self.df['values'].div(df['values'].sum(axis=1), axis=0)
        df['returns'] = (df['values'].sum(axis=1)).pct_change(1).fillna(0)

    def rebalance(self, date):
        """Rebalance the strategy to its initial weighting at a given point in time. The strategy will then hold the
        instruments until it is rebalanced again.

        :param date: The date at which the strategy is to be rebalanced.
        :type date: pandas.Timestamp
        :return: A pandas series containing the values that were rebalanced for each traded instrument
        """
        eod_values = self.df.shift(1).loc[date, 'values'].mul(1 + self.tc.instrument_returns.loc[date, 'daily'])
        eod_portfolio_value = sum(eod_values.values)

        previous_values = self.df.loc[date, 'values'].copy()
        position_value = self.target_weights.mul(eod_portfolio_value)
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
        """Main function to implement trading strategy. The trigger_function has to accept two arrays, the current
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
        # When the trading is finished, we update the daily returns of the portfolio
        self.df['returns'] = self.df['values'].sum(axis=1).pct_change(1).fillna(0)

    def summary_stats(self):
        """Return a series containing the summary statistics for a strategy.

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
        """Save the strategy to 4 separate CSVs. The CSVs contain data on the strategy values, allocations, returns and
        trades. The CSVs are saved in the `datasets` folder.

        :return: None
        """
        path = partial(os.path.join, 'datasets')
        save_name = self.name.lower().replace(' ', '_')
        self.df['values'].sum(axis=1).to_csv(path('{0}_values.csv'.format(save_name)))
        self.df['allocations'].to_csv(path('{0}_allocations.csv'.format(save_name)))
        self.df['returns'].to_csv(path('{0}_returns.csv'.format(save_name)))
        self.trades.to_csv(path('{0}_trades.csv'.format(save_name)))

    def returns_chart(self, ax, color):
        """Produces a line chart of the daily returns.

        :param ax: The Matplotlib axes on which to plot the chart.
        :type ax: matplotlib.axes.Axes
        :param color: The color of the line
        :type color: string
        :return: None
        """
        self.df['returns'].plot(
            ax=ax,
            figsize=(12, 6),
            title='Daily returns of {name} strategy'.format(name=self.name),
            color=color
        )
        ax.set_xlim(self.tc.dates[0], self.tc.dates[-1])

    def returns_distribution_chart(self, ax, color):
        """Produces a Seaborn distplot of the daily returns.

        :param ax: The Matplotlib axes on which to plot the chart.
        :type ax: matplotlib.axes.Axes
        :param color: The color of the distplot
        :type color: string
        :return: None
        """
        sns.distplot(
            a=self.df['returns'],
            color=color,
            ax=ax,
            bins=50,
            label='{name}'.format(name=self.name)
        )

    def asset_allocations_chart(self, ax, cm):
        """Produces an area chart of the asset allocation of the Strategy.

        :param ax: The Matplotlib axes on which to plot the chart.
        :type ax: matplotlib.axes.Axes
        :param cm: The colormap for the areas
        :type cm: matplotlib.colors.Colormap
        :return: None
        """
        self.df['allocations'].plot(
            ax=ax,
            figsize=(12, 6),
            title='Weight of portfolio assets of {name} strategy'.format(name=self.name),
            kind='area',
            legend=False,
            ylim=(0, 1),
            colormap=cm
        )
        ax.set_xlim(self.tc.dates[0], self.tc.dates[-1])
        # Setting y the y labels to percentages
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    def value_chart(self, ax, color):
        """Produces a line chart of the value of the Strategy.

        :param ax: The Matplotlib axes on which to plot the chart.
        :type ax: matplotlib.axes.Axes
        :param color: The color of the line
        :type color: string
        :return: None
        """
        self.df['values'].sum(axis=1).plot(
            ax=ax,
            figsize=(12, 6),
            title='Portfolio values',
            label='Value of {name} strategy'.format(name=self.name),
            color=color
        )
        ax.set_xlim(self.tc.dates[0], self.tc.dates[-1])
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def trades_chart(self, ax, cm):
        """Produces a line chart of the cumulative flows to each instrument of the Strategy.

        :param ax: The Matplotlib axes on which to plot the chart.
        :type ax: matplotlib.axes.Axes
        :param cm: The colormap for the lines
        :type cm: matplotlib.colors.Colormap
        :return: None
        """
        self.trades.cumsum().reindex(self.tc.dates, method='ffill').plot(
            ax=ax,
            figsize=(12, 6),
            title='Cumulative investment per instrument',
            cmap=cm
        )
        ax.set_xlim(self.tc.dates[0], self.tc.dates[-1])
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
