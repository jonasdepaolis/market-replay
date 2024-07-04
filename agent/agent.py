# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# use relative imports for other modules 
from env.market import MarketState, Order, Trade
from env.replay import Backtest # use timestamp_global

# general imports
import abc
import pandas as pd
import textwrap

class BaseAgent(abc.ABC):

    def __init__(self, name):
        """
        Trading agent base class. Subclass BaseAgent to define how a concrete
        Agent should act given different market situations.

        :param name:
            str, agent name
        """

        # agent has market access via market_interface instance
        self.market_interface = MarketInterface(
            exposure_limit=1e6, # ...
            latency=10, # in us (microseconds) 
            transaction_cost_factor=1e-3, # 10 bps
        )

        # ...
        self.name = name

    # event management ---

    @abc.abstractmethod
    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for ten levels
        """

        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def on_trade(self, market_id:str, trade_state:pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trade_state:
            pd.Series, including timestamp, price, quantity
        """

        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def on_time(self, timestamp:pd.Timestamp, timestamp_next:pd.Timestamp):
        """
        This method is called with every iteration and provides the timestamps
        for both current and next iteration. The given interval may be used to
        submit orders before a specific point in time.

        :param timestamp:
            pd.Timestamp, timestamp recorded in this iteration
        :param timestamp_next:
            pd.Timestamp, timestamp recorded in next iteration
        """

        raise NotImplementedError("To be implemented in subclass.")

    def __str__(self):
        """
        String representation.
        """

        # read global timestamp from Backtest class attribute
        timestamp_global = Backtest.timestamp_global

        # string representation
        string = f"""
        ---
        timestamp:      {timestamp_global} (+{self.market_interface.latency} ms)
        ---
        exposure:       {self.market_interface.exposure_total}
        pnl_realized:   {self.market_interface.pnl_realized_total}
        pnl_unrealized: {self.market_interface.pnl_unrealized_total}
        ---
        """

        return textwrap.dedent(string)

    def reset(self):
        """
        Reset agent. 
        """

        # ...
        return self.__init__(self.name)

class MarketInterface: 

    def __init__(self,
        exposure_limit:float=1e6,
        latency:int=10, # in us (microseconds) 
        transaction_cost_factor:float=1e-3, # 10 bps
    ):
        """
        The market interface is used to interact with the market, that is, 
        using the following methods ...

        - `submit_order`: ...
        - `cancel_order`: ...
        - `get_filtered_orders`: ...
        - `get_filtered_trades`: ...
        
        ... to submit and cancel specific orders. 

        :param latency:
            int, latency before order submission (in us), default is 10
        :param transaction_cost_factor:
            float, transcation cost factor per trade (in bps), default is 10
        """

        # containers for related class instances
        self.market_state_list = MarketState.instances
        self.order_list = Order.history
        self.trade_list = Trade.history

        # settings
        self.exposure_limit = exposure_limit # ...
        self.latency = latency # in microseconds ("U"), used only in submit method
        self.transaction_cost_factor = transaction_cost_factor # in bps

    # order management ---

    def submit_order(self, market_id, side, quantity, limit=None):
        """
        Submit market order, limit order if limit is specified.

        Note that, for convenience, this method also returns the order
        instance that can be used for cancellation.

        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares ordered
        :param limit:
            float, limit price to consider, optional
        :return order:
            Order, order instance
        """

        # submit order
        order = Order(
            timestamp=Backtest.timestamp_global + pd.Timedelta(self.latency, "us"), # microseconds
            market_id=market_id,
            side=side,
            quantity=quantity,
            limit=limit,
        )

        return order

    def cancel_order(self, order):
        """
        Cancel an active order.

        :param order:
            Order, order instance
        """

        # cancel order
        order.cancel()

    # order assertion ---

    def _assert_exposure(self, market_id, side, quantity, limit):
        """
        Assert agent exposure. Note that program execution is supposed to
        continue.
        """

        # first, assert that market exists
        assert market_id in self.market_state_list, \
            "market_id '{market_id}' does not exist".format(
                market_id=market_id,
            )

        # calculate position value for limit order
        if limit:
            exposure_change = quantity * limit
        # calculate position value for market order (estimated)
        else:
            exposure_change = quantity * self.market_state_list[market_id].mid_point

        # ...
        exposure_test = self.exposure.copy() # isolate changes
        exposure_test[market_id] = self.exposure[market_id] + exposure_change * {
            "buy": + 1, "sell": - 1,
        }[side]
        exposure_test_total = round(
            sum(abs(exposure) for _, exposure in exposure_test.items()), 3
        )

        # ...
        assert self.exposure_limit >= exposure_test_total, \
            "{exposure_change} exceeds exposure_left ({exposure_left})".format(
                exposure_change=exposure_change,
                exposure_left=self.exposure_left,
            )

    # filtered orders, trades ---

    def get_filtered_orders(self, market_id=None, side=None, status=None):
        """
        Filter Order.history based on market_id, side and status.

        :param market_id:
            str, market identifier, optional
        :param side:
            str, either 'buy' or 'sell', optional
        :param status:
            str, either 'ACTIVE', 'FILLED', 'CANCELLED' or 'REJECTED', optional
        :return orders:
            list, filtered Order instances
        """

        orders = self.order_list

        # orders must have requested market_id
        if market_id:
            orders = filter(lambda order: order.market_id == market_id, orders)
        # orders must have requested side
        if side:
            orders = filter(lambda order: order.side == side, orders)
        # orders must have requested status
        if status:
            orders = filter(lambda order: order.status == status, orders)

        return list(orders)

    def get_filtered_trades(self, market_id=None, side=None):
        """
        Filter Trade.history based on market_id and side.

        :param market_id:
            str, market identifier, optional
        :param side:
            str, either 'buy' or 'sell', optional
        :return trades:
            list, filtered Trade instances
        """

        trades = self.trade_list

        # trades must have requested market_id
        if market_id:
            trades = filter(lambda trade: trade.market_id == market_id, trades)
        # trades must have requested side
        if side:
            trades = filter(lambda trade: trade.side == side, trades)

        return list(trades)

    # symbol, agent statistics ---

    @property
    def exposure(self, result={}):
        """
        Current net exposure that the agent has per market, based statically
        on the entry value of the remaining positions.

        Note that a positive and a negative value indicate a long and a short
        position, respectively.

        :return exposure:
            dict, {<market_id>: <exposure>, *}
        """

        for market_id, _ in self.market_state_list.items():
            
            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell

            # case 1: buy side surplus
            if quantity_unreal > 0:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                result_market = quantity_unreal * vwap_buy
            # case 2: sell side surplus
            elif quantity_unreal < 0:
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = quantity_unreal * vwap_sell
            # case 3: all quantity is realized
            else:
                result_market = 0

            result[market_id] = round(result_market, 3)

        return result

    @property
    def exposure_total(self):
        """
        Current net exposure that the agent has across all markets, based on
        the net exposure that the agent has per market.

        Note that we use the absolute value for both long and short positions.

        :return exposure_total:
            float, total exposure across all markets
        """

        result = sum(abs(exposure) for _, exposure in self.exposure.items())
        result = round(result, 3)

        return result

    @property
    def pnl_realized(self, result={}):
        """
        Current realized PnL that the agent has per market.

        :return pnl_realized:
            dict, {<market_id>: <pnl_realized>, *}
        """
        
        for market_id, _ in self.market_state_list.items():

            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_real = min(quantity_buy, quantity_sell)

            # case 1: quantity_real is 0
            if not quantity_real:
                result_market = 0
            # case 2: quantity_real > 0
            else:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = quantity_real * (vwap_sell - vwap_buy)

            result[market_id] = round(result_market, 3)

        return result

    @property
    def pnl_realized_total(self):
        """
        Current realized pnl that the agent has across all markets, based on
        the realized pnl that the agent has per market.

        :return pnl_realized_total:
            float, total realized pnl across all markets
        """

        result = sum(pnl for _, pnl in self.pnl_realized.items())
        result = round(result, 3)

        return result

    @property
    def pnl_unrealized(self, result={}):
        """
        This method returns the unrealized PnL that the agent has per market.

        :return pnl_unrealized:
            dict, {<market_id>: <pnl_unrealized>, *}
        """

        for market_id, market in self.market_state_list.items():

            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell

            # case 1: buy side surplus
            if quantity_unreal > 0:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy 
                result_market = abs(quantity_unreal) * (market.best_bid - vwap_buy)
            # case 2: sell side surplus
            elif quantity_unreal < 0:
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = abs(quantity_unreal) * (vwap_sell - market.best_ask)
            # case 3: all quantity is realized
            else:
                result_market = 0

            result[market_id] = round(result_market, 3)

        return result

    @property
    def pnl_unrealized_total(self):
        """
        Current unrealized pnl that the agent has across all markets, based on
        the unrealized pnl that the agent has per market.

        :return pnl_unrealized_total:
            float, total unrealized pnl across all markets
        """

        result = sum(pnl for _, pnl in self.pnl_unrealized.items())
        result = round(result, 3)

        return result

    @property
    def exposure_left(self):
        """
        Current net exposure left before agent exceeds exposure_limit.

        :return exposure_left:
            float, remaining exposure
        """

        # TODO: include self.pnl_realized_total?
        result = self.exposure_limit - self.exposure_total
        result = round(result, 3)

        return result

    @property
    def transaction_cost(self):
        """
        Current trading cost based on trade history, accumulated throughout
        the entire backtest.

        :transaction_cost:
            float, accumulated transaction cost
        """

        result = sum(t.price * t.quantity for t in self.trade_list)
        result = result * self.transaction_cost_factor
        result = round(result, 3)

        return result


