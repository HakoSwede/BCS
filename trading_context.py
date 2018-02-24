import pandas as pd


class TradingContext:
    """
    Data structure for storing various parameters used in trading a strategy.

    The trading context stores the dates for the backtest, the instruments to be traded, the returns of those
    instruments, the starting cash, and the commission per trade.

    :param dates: A datetime index of the dates for which the strategy will be backtested
    :type dates: pandas.DatetimeIndex
    :param tickers: A list of the ticker symbols of the traded instruments
    :type tickers: list
    :param instrument_returns: A dataframe containing the returns of the traded instruments
    :type instrument_returns: pandas.DataFrame
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
        self._add_cumulative_returns(instrument_returns)

    def _add_cumulative_returns(self, instrument_returns):
        """
        Adds cumulative returns to the instrument_returns dataframe
        :param instrument_returns: Dataframe containing the daily returns of the instruments.
        :return: None
        """
        instrument_returns.columns = pd.MultiIndex.from_product([['daily'], self.tickers])
        instrument_returns[list(zip(['cumulative'] * 8, self.tickers))] = (instrument_returns['daily'] + 1).cumprod()
