# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# specific imports
from decimal import Decimal

# general imports
import copy
import numpy as np
import pandas as pd


class MarketState:

    instances = dict() # instance store

    def __init__(self, market_id):
        """
        Market state is implemented as a stateful object in order to ensure
        price-time-priority in simulated order execution. This means that 
        liquidity denoted by (<timestamp>, <quantity>) is added to and removed 
        from the price levels available in the market state. There are two 
        states that are continually updated ...

        - `_posttrade_state` (post-trade, consistent with historical data)
        - `_pretrade_state_bid`, `_pretrade_state_ask` (pre-trade, temporary)

        ... that reflect the post-trade market state (based on original data)
        and pre-trade market state (used to match agent orders against),
        respectively. In order to manipulate these states, this class implements
        two methods ...

        - `update(book_state, trade_state)`
        - `match()`
        
        Note that this model of a market state requires a set of assumptions, 
        especially due to the nature of level 2 data:

        - orders submitted by the agent do NOT have market impact!
        - liquidity is added to the end of the liquidity_list
        - liquidity is assumed to always be taken from the start of the 
        liquidity_list 
        - ...

        All market instances are stored in and may be accessed through the
        `instances` class attribute (dictionary).

        :param market_id:
            str, market identifier
        """

        # static attributes from arguments
        self.market_id = market_id

        # global attributes update
        self.__class__.instances.update({market_id: self})

    # properties ---
    
    @property
    def timestamp(self):
        """
        `timestamp` is based on the most recent book update. 
        """
        
        # ...
        try:
            timestamp = self._timestamp
        except:
            timestamp = None
        
        # based on current market state, return timestamp
        return timestamp
    
    @property
    def state(self):
        """
        `state` represents the most recent market post-trade market state that
        would be observable by a market participant ('pseudo' level 3).
        
        It is provided as a tuple of dictionaries with the following structure
        (
            {<price>: [(<timestamp>, <quantity>), *], *}, # bid side
            {...}, # ask side
        )
        """
        
        # ...
        try:
            state = (self._posttrade_state_bid, self._posttrade_state_ask)
        except:
            state = None

        # based on current market state, return entire post-trade dictionary
        return state

    @property
    def midpoint(self):
        """
        `midpoint` is based on the most recent book update. 
        """
        
        # ...
        try:
            midpoint = self._midpoint_this
        except:
            midpoint = None
        
        # based on current market state, return midpoint
        return midpoint

    @property
    def best_bid(self):
        """
        `best_bid` is based on the most recent book update. 
        """

        # ...
        try:
            best_bid = max(k for k, v in self._posttrade_state_bid.items()
                if len(v) > 0
            )
        except:
            best_bid = None
        
        # based on current market state, return best_bid
        return best_bid

    @property
    def best_ask(self):
        """
        `best_ask` is based on the most recent book update. 
        """

        # ...
        try:
            best_ask = min(k for k, v in self._posttrade_state_ask.items()
                if len(v) > 0
            )
        except:
            best_ask = None
        
        # based on current market state, return best_ask
        return best_ask

    @property
    def tick_size(self): # infer tick_size at runtime
        """
        `tick_size` is inferred dynamically at runtime based on the most recent
        book update, identifying the greatest common divisor among all inluded 
        price levels. 
        """

        # extract all price levels present in the book update
        tick_size = np.array(list(self._book_this)) * 1e3
        # tick_size is greatest common divisor among price levels
        tick_size = np.gcd.reduce(
            np.around(tick_size).astype(int)
        ) / 1e3

        return tick_size

    # update ---

    def update(self, book_update, trade_update):
        """
        Update the market state that is represented by two separate stages,
        post-trade state and pre-trade state. 
        
        (Post-trade state) Use book_update (mandatory) to infer from the 
        aggregated quantity provided in the level 2 market data the individual 
        orders standing on each price level. The post-trade state is updated
        continuously with every update. 

        (Pre-trade state) Use trade_update (optional) to eliminate the 
        incluence that one or multiple trades have on the post-trade state. 
        The pre-trade state is used to match simulated orders against. 

        Main methods are ...
        - `_update_posttrade_state()`: ...
        - `_update_pretrade_state()`: ...
        - `_update_simulated_orders()`: ...

        Helper methods are ...
        - `_add_liquidity(...)`: add liquidity to a given price level
        - `_use_liquidity(...)`: remove liquidity from a given price level
        - `_restore_liquidity(...)`: restore liquidity for a given price level

        Note that this implementation does not directly model both sides of the 
        market, but we keep track of the midpoint to separate bid and ask side. 

        :param book_update:
            pd.Series, book data
        :param trade_update:
            pd.Series, trade data, aggregated per timestamp
        """

        # unpack pd.Series into list for each book update and trade update
        timestamp, *book_update = book_update.values
        _, *trade_update = trade_update.values # optional (may be empty pd.Series)

        # ensure that each ask is larger than its respective bid
        is_corrupted = any(bid >= ask 
            for bid, ask in zip(book_update[0::4], book_update[2::4])
        )
        # otherwise, skip this particular update
        if is_corrupted:
            return

        # set dictionary representation for time t-1 (_book_last)
        if hasattr(self, "_book_this"): 
            self._book_last = self._book_this 
        # if variable does not yet exist, start with empty dictionary
        else:
            self._book_last = dict()
        
        # set dictionary representation for book update at time t (_book_this)
        self._book_this = dict(zip(book_update[0::2], book_update[1::2]))
        # set nested list representation for trade update at time t (_trade_this)
        self._trade_this = trade_update

        # set variables required to determine current state
        self._timestamp = timestamp
        self._midpoint_last = sum(list(self._book_last)[:2]) / 2 # [<L1-BidPrice>, <L1-AskPrice>]
        self._midpoint_this = sum(list(self._book_this)[:2]) / 2 # ...

        # if variable does not exist, set empty dictionary for post-trade state
        if not hasattr(self, "_posttrade_state"): 
            self._posttrade_state = dict() 

        # create deepcopy of post-trade state to later reconstruct timestamps in pre-trade state
        self._SNAPSHOT = copy.deepcopy(self._posttrade_state)        

        # run update on post-trade state 
        _ = self._update_posttrade_state()
        
        # run update on pre-trade state
        _ = self._update_pretrade_state()
        
        # fetch the relevant orders submitted by the trading agent
        _ = self._update_simulated_orders()

    def _update_posttrade_state(self):
        """
        Compute post-trade state that is identical to the historical book
        state, but that keeps track of the detailed liquidity_list per price 
        level. The post-trade state resembles the internal state that is
        continuously being updated over time. 

        The post-trade state data structure is ...
        {<price>: [(<timestamp>, <quantity>), *], *}
        """        

        # book_difference, [(<price>, <quantity>), *]
        book_difference = []

        # test whether price is on opposite sides of last and this midpoint 
        same_side = lambda price: (
            (self._midpoint_this - price) * (self._midpoint_last - price)
        ) > 0

        # compute book_difference
        for price in set(self._book_this) | set(self._book_last):
            # price on same side: remove (this - last) quantity
            if same_side(price):
                book_difference.append(
                    (price, self._book_this.get(price, 0) - self._book_last.get(price, 0))
                )
            # price on opposite side: remove last quantity, add this quantity
            else:
                book_difference.append( # remove
                    (price, self._book_last.get(price, 0) * (-1))
                )
                book_difference.append( # add
                    (price, self._book_this.get(price, 0))
                )

        # apply book_difference
        for price, qdiff in book_difference:
            # if positive qdiff: add liquidity to a given price level
            if qdiff > 0:
                self._posttrade_state[price] = self._add_liquidity(
                    liquidity_list=self._posttrade_state.get(price, []),
                    timestamp=self._timestamp, quantity=abs(qdiff),
                )
            # if negative qdiff: use liquidity from a given price level
            elif qdiff < 0:
                self._posttrade_state[price] = self._use_liquidity(
                    liquidity_list=self._posttrade_state.get(price, []),
                    quantity=abs(qdiff),
                )
            # ...
            else:
                pass

        # NOTE: the post-trade state combines both sides given midpoint context ...

        # set post-trade state (bid and ask together)
        self._posttrade_state = self._posttrade_state

        # NOTE: ... but is additionally split by side for further processing

        # set post-trade state, bid side with sorted price levels (DESCENDING)
        self._posttrade_state_bid = {p: q for p, q 
            in sorted(self._posttrade_state.items(), reverse=True)
            if p < self._midpoint_this
        }
        self._posttrade_state_bid = self._posttrade_state_bid

        # set post-trade state, ask side with sorted price levels (ASCENDING)
        self._posttrade_state_ask = {p: q for p, q 
            in sorted(self._posttrade_state.items(), reverse=False)
            if p > self._midpoint_this
        }
        self._posttrade_state_ask = self._posttrade_state_ask
    
    def _update_pretrade_state(self):
        """
        Compute pre-trade state that is the post-trade state without the 
        influence of the historical trade state, meaning that any trades are 
        reverted as if they never happened. The pre-trade state is computed 
        with every update, there is no continuity as with the post-trade state 
        since simulated orders are not designed to have market impact. 
        
        (case 1) book update without trade update: post-trade state and 
        pre-trade state are identical

        (case 2) book update with trade update: pre-trade state includes 
        additional liquidity competing with the agent orders, only at that 
        point in time

        The pre-trade state is provided separately for each side.

        The pre-trade state data structure is ...
        {<price>: [(<timestamp>, <quantity>), *], *}
        """
        
        # create deepcopy of post-trade state (bid side) as a basis to compute pre-trade state
        self._pretrade_state_bid = copy.deepcopy(self._posttrade_state_bid) 
        # ...
        self._pretrade_state_ask = copy.deepcopy(self._posttrade_state_ask)

        # check that trade_state is a nested list, as otherwise it must be empty
        require_revert = isinstance(self._trade_this[0], list) 

        # if there exists a valid trade_state, run pre-trade reversion steps
        if require_revert: 
            for price, quantity in zip(*self._trade_this):

                # assign roles side_1st (standing side), side_2nd (matching side) 
                if price < self._midpoint_last:
                    side_1st = self._pretrade_state_bid # bid was standing (1st)
                    side_2nd = self._pretrade_state_ask # ask was matching (2nd)
                # ...
                if price > self._midpoint_last:
                    side_1st = self._pretrade_state_ask # ...
                    side_2nd = self._pretrade_state_bid # ...

                # standing side (1): restore liquidity (t-1), use original timestamp(s)
                side_1st[price], surplus = self._restore_liquidity(
                    liquidity_list=side_1st.get(price, []), 
                    liquidity_list_init=self._SNAPSHOT.get(price, []), 
                    quantity=quantity,
                )
                # standing side (2): add liquidity (t), use current timestamp, only in case of surplus
                side_1st[price] = self._add_liquidity(
                    liquidity_list=side_1st.get(price, []),
                    timestamp=self._timestamp, 
                    quantity=surplus,
                )
                # matching side (1): add liquidity (t), use current timestamp
                side_2nd[price] = self._add_liquidity(
                    liquidity_list=side_2nd.get(price, []),
                    timestamp=self._timestamp, 
                    quantity=quantity,
                )

        # otherwise, pre-trade state remains identical to post-trade state
        else:
            pass

        # NOTE: the pre-trade state can only exist for each individual side due to potential crossing

        # set pre-trade state, bid side with sorted price levels (DESCENDING)
        self._pretrade_state_bid = dict(
            sorted(self._pretrade_state_bid.items(), reverse=True)
        )
        self._pretrade_state_bid = self._pretrade_state_bid

        # set pre-trade state, ask side with sorted price levels (ASCENDING)
        self._pretrade_state_ask = dict(
            sorted(self._pretrade_state_ask.items(), reverse=False)
        )
        self._pretrade_state_ask = self._pretrade_state_ask

    def _update_simulated_orders(self):
        """
        View based on Order.history, includes all AGENT orders filtered by
        status 'ACTIVE', older-than-current timestamp (with regard to 
        latency), and corresponding market_id. 

        The simulated orders are provided separately for each side, sorted 
        according to price-time-priority, respectively. 
        """

        # orders must have status 'ACTIVE'
        orders = filter(lambda order: order.status == "ACTIVE", Order.history)
        # orders must have corresponding market_id 
        orders = filter(lambda order: order.market_id == self.market_id, orders)
        # orderst must have timestamp greater than current timestamp
        orders = filter(lambda order: order.timestamp <= self._timestamp, orders)
        # important: cast iterable into list! 
        orders = list(orders) 

        # filter buy orders
        orders_buy = filter(lambda order: order.side == "buy", orders)
        # sort by (1) limit DESCENDING and (2) time ASCENDING
        orders_buy = sorted(orders_buy, key=lambda x: x.timestamp)
        orders_buy = sorted(orders_buy, 
            key=lambda x: x.limit or min(self._posttrade_state_ask), reverse=True
        )
        self._orders_buy = orders_buy

        # filter sell orders
        orders_sell = filter(lambda order: order.side == "sell", orders)
        # sort by (1) limit ASCENDING and (2) time ASCENDING
        orders_sell = sorted(orders_sell, key=lambda x: x.timestamp)
        orders_sell = sorted(orders_sell, 
            key=lambda x: x.limit or max(self._posttrade_state_bid), reverse=False
        )
        self._orders_sell = orders_sell

    # update helper methods ---

    @staticmethod
    def _add_liquidity(liquidity_list, timestamp, quantity):
        """
        Add liquidity to a given liquidity_list, that is a partial quantity
        tagged with its corresponding timestamp.

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level
        :param timestamp:
            pd.Timestamp, timestamp to add
        :param quantity:
            int, liquidity to add
        :return liquidity_list:
            list, (timestamp, quantity) tuples + added liquidity
        """

        # bypass in case of empty quantity
        if (not quantity):
            return liquidity_list

        # convert to dictionary, timestamps are unique
        liquidity = dict(liquidity_list)
        # aggregate added quantity with pre-existent quantity
        liquidity[timestamp] = liquidity.get(timestamp, 0) + quantity
        # convert to list of tuples
        liquidity_list = liquidity.items()

        # sort by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list

    @staticmethod
    def _use_liquidity(liquidity_list, quantity):
        """
        Use liquidity from a given liquidity_list, starting with the quantity
        tagged with the oldest available timestamp.

        Note that quantity will never exceed liquidity. There are two cases to
        consider ...
        - self.update: cannot happen in historical data
        - self.match: controls for using more than what is available

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level
        :param quantity:
            int, quantity to use from liquidity_list
        :return liquidity_list:
            list, (timestamp, quantity) tuples - used liquidity
        """

        # bypass in case of empty quantity or empty liquidity_list
        if (not quantity) or (not liquidity_list):
            return liquidity_list

        # determine used liquidity
        timestamp_list, quantity_list = zip(*liquidity_list)
        quantity_cumsum = np.cumsum(quantity_list)
        i = np.argwhere(quantity_cumsum >= quantity).flatten()[0]
        liquidity_list = liquidity_list[i+1:]

        # determine partial liquidity to prepend
        timestamp = timestamp_list[i]
        remainder = quantity_cumsum[i] - quantity
        insert = (timestamp, remainder)

        # prepend (timestamp, quantity_left) to liquidity_list
        if remainder:
            liquidity_list.insert(0, insert)

        # sort by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list

    @staticmethod
    def _restore_liquidity(liquidity_list, liquidity_list_init, quantity):
        """
        Restore liquidity less than or equal to the liquidity used between
        last state (liquidity_list_init) and this state (liquidity_list).

        Other than _add_liquidity, _restore_liquidity includes the initial
        timestamps and is preprended to the liquidity_list.

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level (t)
        :param liquidity_list_init:
            list, (timestamp, quantity) tuples for a given price level (t-1)
        :param quantity:
            int, quantity to restore
        :return liquidity_list:
            list, (timestamp, quantity) tuples + restored liquidity
        :return quantity:
            int, remaining quantity surplus
        """

        # convert to dictionary, timestamps are unique
        liquidity = dict(liquidity_list)
        liquidity_init = dict(liquidity_list_init)

        # ...
        for timestamp in sorted(liquidity_init):
            difference = max(
                liquidity_init.get(timestamp, 0) - liquidity.get(timestamp, 0),
                0 # proceed only if quantity_init (t-1) >= quantity (t)
            )
            restored = min(
                quantity, # remaining quantity
                difference # difference that can be restored
            )
            liquidity[timestamp] = liquidity.get(timestamp, 0) + restored
            quantity -= restored

        # convert to list of tuples
        liquidity_list = liquidity.items()

        # sort liquidity_list by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list, quantity

    # match ---

    def match(self):
        """
        Match simulated standing buy orders against pre-trade ask state, and 
        simulated standing sell orders against pre-trade bid state. Iterate
        the order queue per side and level, matching each order individually.
        """

        # pretrade_state_ask is consumed, COPY_STATE_ASK is competing state
        PRETRADE_STATE_ASK_FIXED = copy.deepcopy(self._pretrade_state_ask)
        pretrade_state_ask = self._pretrade_state_ask
        
        # pretrade_state_bid is consumed, COPY_STATE_BID is competing state
        PRETRADE_STATE_BID_FIXED = copy.deepcopy(self._pretrade_state_bid)
        pretrade_state_bid = self._pretrade_state_bid

        # match agent buy orders against ask state, bid state is competing
        for order in self._orders_buy:
            pretrade_state_ask = self._match_order(
                order=order, 
                state=pretrade_state_ask, 
                state_compete=PRETRADE_STATE_BID_FIXED,
            )

        # match agent sell orders against bid state, ask state is competing
        for order in self._orders_sell:
            pretrade_state_bid = self._match_order(
                order=order, 
                state=pretrade_state_bid, 
                state_compete=PRETRADE_STATE_ASK_FIXED,
            )

    def _match_order(self, order, state, state_compete):
        """
        Match a single order, that can be either a market order or a limit 
        order. 
        
        (market order) does specify a limit, uses all available liquidity

        (limit order) does not specify a limit, uses only liquidity 
        given at price levels better than (or equal to) the specified order 
        limit.
        
        Note that this method receives only order, state (opposite side as 
        order), and state_compete (same side as order). Longer-standing 
        liquidity on the competing side is given priority over agent order.

        :param order:
            Order, order instance with side corresponding to state
        :param state:
            dict, state filtered by side corresponding to order, gets consumed
        :param state_compete:
            dict, state filtered by side competing with order, remains fixed
        :return state:
            dict, state after order excecution
        """

        # select operator to understand if price is 'better' than limit
        better_than = {
            "buy": np.less_equal,
            "sell": np.greater_equal,
        }[order.side]    

        # ...
        for price, liquidity_list in state.items():

            # break matching algorithm when price is worse than limit
            if order.limit and not better_than(price, order.limit):
                break

            # determine how much quantity can be used by agent order
            quantity_available = sum(q for _, q in liquidity_list)
            quantity_blocked = sum(q for t, q in state_compete.get(price, [])
                if t <= order.timestamp # standing orders are prioritized
            )
            quantity_available = max(0, quantity_available - quantity_blocked)
            quantity_used = min(quantity_available, order.quantity_left)

            # execute (partial) order at this price level
            if quantity_used:
                order.execute(self._timestamp, quantity_used, price)

            # use liquidity
            state[price] = self._use_liquidity(
                liquidity_list=state[price], 
                quantity=quantity_used,
            )

        return state

    # class method ---

    @classmethod
    def reset_instances(class_reference):
        """
        Reset all class instances, that is, simply replace the old MarketEngine 
        instance with a new one so that we do not need to reset everything 
        manually. 
        """

        # delete all elements in MarketState.instances (dictionary)
        class_reference.instances.clear() 


class Order:

    history = list() # instance store

    def __init__(self, timestamp, market_id, side, quantity, limit=None):
        """
        Instantiate order.

        Note that an order can have different statuses:
        - 'ACTIVE': default
        - 'FILLED': set in Order.execute when there is no quantity left
        - 'CANCELLED': set in Order.cancel
        - 'REJECTED': set in Order.__init__

        Note that all order instances are stored in and may be accessed through
        the `history` class attribute (list).

        :param timestamp:
            pd.Timestamp, date and time that order was submitted
        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares ordered
        :param limit:
            float, limit price to consider, optional
        """

        # static attributes from arguments
        self.timestamp = timestamp
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.limit = limit
        self.order_id = len(self.__class__.history)

        # dynamic attributes
        self.quantity_left = quantity
        self.status = "ACTIVE"
        self.related_trades = []

        # assert order parameters
        try:
            self._assert_params()
        # set status 'REJECTED' if parameters are invalid
        except Exception as error:
            print("(INFO) order {order_id} was rejected: {error}".format(
                order_id=self.order_id,
                error=error,
            ))
            self.status = "REJECTED"
        # ...
        else:
            print("(INFO) order {order_id} was accepted: {self}".format(
                order_id=self.order_id,
                self=self,
            ))

        # global attributes update
        self.__class__.history.append(self)

    def _assert_params(self):
        """
        Assert order parameters and provide information about an erroneous
        order submission. Note that program execution is supposed to continue.
        """

        # first, assert that market exists
        assert self.market_id in MarketState.instances, \
            "market_id '{market_id}' does not exist".format(
                market_id=self.market_id,
            )
        # assert that market_state is available
        timestamp = MarketState.instances[self.market_id].timestamp
        assert not pd.isnull(timestamp), \
            "trading is yet to start for market '{market_id}'".format(
                market_id=self.market_id
            )
        # assert that side is valid
        assert self.side in ["buy", "sell"], \
            "side can only take values 'buy' and 'sell', not '{side}'".format(
                side=self.side,
            )
        # assert that quantity is valid
        assert float(self.quantity).is_integer(), \
            "quantity can only take integer values, received {quantity}".format(
                quantity=self.quantity,
            )
            
        # the remaining asserts are required only in case of a limit order
        if not self.limit:
            return
        
        # assert that limit is valid
        tick_size = MarketState.instances[self.market_id].tick_size
        assert not Decimal(str(self.limit)) % Decimal(str(tick_size)), \
            "limit {limit} is too granular for tick_size {tick_size}".format(
                limit=self.limit,
                tick_size=tick_size,
            )

    def execute(self, timestamp, quantity, price):
        """
        Execute order.

        Note that an order is split into multiple trades if it is matched
        across multiple prices levels.

        :param timestamp:
            pd.Timestamp, date and time that order was executed
        :param quantity:
            int, matched quantity
        :param price:
            float, matched price
        """

        # execute order (partially)
        trade = Trade(timestamp, self.market_id, self.side, quantity, price)
        self.related_trades.append(trade)

        # update remaining quantity
        self.quantity_left -= quantity

        # set status 'FILLED' if self.quantity_left is exhausted
        if not self.quantity_left:
            self.status = "FILLED"

    def cancel(self):
        """
        Cancel order.
        """

        # set status 'CANCELLED' if order is still active
        if not self.status in ["CANCELLED, FILLED, REJECTED"]:
            self.status = "CANCELLED"

    def __str__(self):
        """
        String representation.
        """

        string = "{side} {market_id} with {quantity}@{limit}, {time}".format(
            time=self.timestamp,
            market_id=self.market_id,
            side=self.side,
            quantity=self.quantity,
            limit=self.limit or 'market',
        )

        return string

    @classmethod
    def reset_history(class_reference):
        """
        Reset order history.
        """
        
        # delete all elements in Order.history (list)
        del class_reference.history[:]


# TODO
class OrderPool: 
    
    def __init__(self):    
        pass 


class Trade:

    history = list() # instance store

    def __init__(self, timestamp, market_id, side, quantity, price):
        """
        Instantiate trade.

        Note that all trade instances are stored in and may be accessed through
        the `history` class attribute (list).

        :param timestamp:
            pd.Timestamp, date and time that trade was created
        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares executed
        :param price:
            float, price of shares executed
        """

        # static attributes from arguments
        self.timestamp = timestamp
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.price = price
        self.trade_id = len(self.__class__.history)

        # ...
        print("(INFO) trade {trade_id} was executed: {self}".format(
            trade_id=self.trade_id,
            self=self,
        ))

        # global attributes update
        self.__class__.history.append(self)

    def __str__(self):
        """
        String representation.
        """

        string = "{side} {market_id} with {quantity}@{price}, {time}".format(
            time=self.timestamp,
            market_id=self.market_id,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
        )

        return string

    @classmethod
    def reset_history(class_reference):
        """
        Reset trade history. 
        """
        
        # delete all elements in Trade.history (list)
        del class_reference.history[:]


# TODO
class TradePool: 
    
    def __init__(self):    
        pass 


