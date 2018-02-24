import numpy as np
import pandas as pd

class TradingContext:
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
    def __init__(self, dates, tickers, instrument_returns, starting_cash, commission):
        self.dates = dates
        self.tickers = tickers
        self.instrument_returns = instrument_returns
        self.starting_cash = starting_cash
        self.commission = commission