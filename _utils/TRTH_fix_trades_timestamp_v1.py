# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

"""
Given below are the columns and dtypes corresponding to the different source
types: 'BOOK', 'TRADES'.

- 'BOOK' is based on TRTH book data, including a timestamp (UTC) and 10 limit
order book levels, each with bid/ask price/size.

- 'TRADES' is based on TRTH trades data, including timestamp (UTC), type
(auction, trade), price and volume.
"""

dtype_book = {
    "TIMESTAMP_UTC": str,
    "L1-BidPrice": float, "L1-BidSize": int, "L1-AskPrice": float, "L1-AskSize": int,
    "L2-BidPrice": float, "L2-BidSize": int, "L2-AskPrice": float, "L2-AskSize": int,
    "L3-BidPrice": float, "L3-BidSize": int, "L3-AskPrice": float, "L3-AskSize": int,
    "L4-BidPrice": float, "L4-BidSize": int, "L4-AskPrice": float, "L4-AskSize": int,
    "L5-BidPrice": float, "L5-BidSize": int, "L5-AskPrice": float, "L5-AskSize": int,
    "L6-BidPrice": float, "L6-BidSize": int, "L6-AskPrice": float, "L6-AskSize": int,
    "L7-BidPrice": float, "L7-BidSize": int, "L7-AskPrice": float, "L7-AskSize": int,
    "L8-BidPrice": float, "L8-BidSize": int, "L8-AskPrice": float, "L8-AskSize": int,
    "L9-BidPrice": float, "L9-BidSize": int, "L9-AskPrice": float, "L9-AskSize": int,
    "L10-BidPrice": float, "L10-BidSize": int, "L10-AskPrice": float, "L10-AskSize": int,
}

dtype_trades = {
    "TIMESTAMP_UTC": str,
    "Type": str, # this is needed only to exclude auctions!
    "Price": float,
    "Volume": int,
}

def _fix_trades_timestamp(
        trades:pd.DataFrame,
        book:pd.DataFrame,
        tolerance:int=200,
        verbose:bool=True,
    ) -> pd.DataFrame:
    """
    Fix inconsistent timestamps in TRTH trades data by replacing it with the
    closest plausible timestamp available in book data, removing those instances
    that exceed the search range. The applied heuristic is described in

    https://arxiv.org/pdf/1604.02759.pdf

    The authors report a plateau in performance at ~200ms, hence we use this
    as default value for the tolerance parameter. For 2016 TRTH data, this
    means 80% - 90% matched trades.

    :param trades:
        pd.DataFrame, trades for a given stock
    :param book:
        pd.DataFrame, book for a given stock
    :param tolerance:
        int, millisecond-based range to search around a trades timestamp, default is 200
    :param verbose:
        bool, print updates
    :return trades:
        pd.DataFrame, trades for a given stock with fixed timestamps
    """

    datetime = "TIMESTAMP_UTC"
    tolerance = pd.Timedelta(tolerance, "milliseconds")

    # filter auctions upfront
    trades = trades.loc[trades["Type"] != "Auction", :]
    trades = trades.reset_index(drop=True)

    # setup counter
    num_matched = 0
    num_total = len(trades)

    # replacement timestamps
    time_repl_list = []

    # filter trades that cannot be matched with book
    for trades_index, trade in trades.iterrows():

        # find all book_states within tolerance of a given trade
        results = book[datetime].between(
            trade[datetime]-tolerance, # lower range limit
            trade[datetime]+tolerance, # upper range limit
        )

        # get indices and ensure that there exists an index i-1
        index_list = results[results].index
        index_list = list(filter(lambda x: x > 0, index_list))

        # candidates for particular replacement timestamp
        time_cand_list = []

        # find all book_states that are plausible for a given trade
        for book_index in index_list:

            # ...
            _, *state_init = book.iloc[book_index-1].values
            time_book, *state_post = book.iloc[book_index].values

            # find differences in book_state between t-1 and t
            book_diff = _book_diff(state_init, state_post)

            # check plausibility: trade volume <= volume difference
            if book_diff.get(trade["Price"], 0) >= trade["Volume"]:
                time_cand_list.append(time_book)

        # find the single book_state that is closest to a given trade
        time_repl = min(time_cand_list,
            key=lambda time_book: abs(time_book - trade[datetime]),
            default=None, # return None if list is empty
        )

        # append the corresponding timestamp to time_repl_list
        time_repl_list.append(time_repl)
        num_matched += bool(time_repl)

        # report success rate up to this point
        if verbose and (trades_index % 100 == 0):
            print("UPDATE: {a}/{b} out of {c} ({rel_success}%)".format(
                a=num_matched, b=trades_index, c=num_total,
                rel_success=round(num_matched/(trades_index+1)*100, 2),
            ))

    # insert timestamp replacements and filter rows where timestamp is NaT
    trades.loc[:, datetime] = time_repl_list
    # delete rows where timestamp is missing
    trades = trades.loc[~trades[datetime].isna(), :]
    # reset index
    trades = trades.reset_index(drop=True)

    return trades

def _book_diff(
        book_last:list,
        book_this:list,
    ) -> dict:
    """
    Given two book states for t-1 and t, determine the differences in quantity
    for each price level so that a positive integer value denotes quantity that
    has been removed (due to execution or cancellation).

    Note that both inputs are lists of form '[BidP, BidS, AskP, AskS] *
    10 levels'.

    :param book_last:
        pd.Series, book state in t-1
    :param book_this:
        pd.Series, book state in t
    :return difference:
        dict, book state difference
    """

    # set dictionary representation for t-1 (book_last), t (book_this)
    book_last = dict(zip(book_last[0::2], book_last[1::2]))
    book_this = dict(zip(book_this[0::2], book_this[1::2]))

    # function to return mid point given book_state as input dictionary (0 if empty)
    mid_point = lambda input: sum(list(input)[:2]) / 2
    # function to test whether price is on opposite sides of last and this mid point
    opposite_side = lambda price: (
        (mid_point(book_this) - price) * (mid_point(book_last) - price)) < 0

    # book_diff, {<price>: <quantity>, *}
    book_diff = dict()
    # input_difference: compute ...
    for price in set(book_this) | set(book_last):
        # price not on opposite sides: difference = (last - this) quantity
        if not opposite_side(price):
            book_diff[price] = book_last.get(price, 0) - book_this.get(price, 0)
        # price on opposite sides: difference = last quantity
        if opposite_side(price):
            book_diff[price] = book_last.get(price, 0)

    return book_diff

if __name__ == "__main__":

    # load trades data
    trades = pd.read_csv(
        "/Volumes/HDD_8TB/_READ_ONLY/TRTH_TRDS/Trades_ADSGn.DE_20160101_20160131.csv",
        usecols=dtype_trades.keys(), dtype=dtype_trades, parse_dates=["TIMESTAMP_UTC"]
    )
    # load book data
    book = pd.read_csv(
        "/Volumes/HDD_8TB/_READ_ONLY/TRTH_BOOK/Books_ADSGn.DE_20160101_20160131.csv",
        usecols=dtype_book.keys(), dtype=dtype_book, parse_dates=["TIMESTAMP_UTC"]
    )
    # fix trades data
    trades_fixed = _fix_trades_timestamp(trades, book, tolerance=200, verbose=True)
