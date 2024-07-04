# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

"""
Given below are the columns and dtypes corresponding to the different source
types: 'TRADES'.

'TRADES' is based on TRTH trades data, including timestamp (UTC), type (auction,
trade), price and volume.
"""

dtype_trades = {
    "TIMESTAMP_UTC": str,
    "Price": float,
    "Volume": int,
}

def _aggregate_trades_timestamp(
        trades:pd.DataFrame,
    ):
    """
    Handle non-unique timestamp-based index. Note that this step is necessary
    so the sources can be properly aligned!

    :param trades:
        pd.DataFrame, trades for a given stock
    :return trades:
        pd.DataFrame, trades for a given stock with aggregated rows
    """

    datetime = "TIMESTAMP_UTC"

    # aggregate multiple rows using a single timestamp with a list of values
    # per column. NOTE: this leads to every value being stored in a list
    trades = (trades
        .groupby(datetime)
        .agg(list)
        .reset_index()
    )
    
    # eliminate rounding errors to 3 decimal places
    trades[["Price", "Volume"]] = (trades[["Price", "Volume"]]
        .apply(lambda row: [[round(value, 3) 
            for value in field] for field in row]
        )
    )

    return trades

if __name__ == "__main__":

    # load trades data
    trades = pd.read_csv(
        "/Volumes/HDD_8TB/_READ_ONLY/TRTH_TRDS/Trades_ADSGn.DE_20140101_20140131.csv",
        usecols=dtype_trades.keys(), dtype=dtype_trades, parse_dates=["TIMESTAMP_UTC"]
    )
    # aggregate trades data
    trades_aggregated = _aggregate_trades_timestamp(trades)
