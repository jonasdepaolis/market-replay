# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from TRTH_aggregate_book_timestamp import _aggregate_book_timestamp
from TRTH_aggregate_trades_timestamp import _aggregate_trades_timestamp
from TRTH_fix_trades_timestamp_v1 import _fix_trades_timestamp

import json
import os
import pandas as pd

"""
Given below are the columns and dtypes corresponding to the different source
types: 'BOOK', 'TRADES'.

- 'BOOK' is based on TRTH book data, including a timestamp (UTC) and 10 limit
order book levels, each with bid/ask price/size.
  -> aggregated: drop duplicates wrt timestamp (keep last)
- 'TRADES' is based on TRTH trades data, including timestamp (UTC), type 
(auction, trade), price and volume.
  -> aggregated: keep duplicates wrt timestamp in a list

IMPORTANT: 
    
- Currently, we aggregate book data in that we drop duplicate rows wrt 
timestamp. 
- Consequently, multiple trades are mapped to the same timestamp in book data.
- Due to order submissions and cancellations that may have occured in the 
deleted book rows, we cannot tell whether all trades that are mapped to a 
single timestamp are actually plausible.
- Since we only check that the difference in quantity per book level (from 
t-1 to t) is greater than the quantity taken by a trade, we could possibly 
allow situations in which too many trades are mapped to a single book 
transition.
- However, the only way that this would affect the backtest is that sometimes
(although rarely) too much liquidity could be made available in the pre-trade 
state. 

TODO (v2):
    
- do NOT aggregate book data and instead improve _fix_trades_timestamp so that 
a group of same-index book rows would be mapped to a group of matching trades 
(not necessarily the same amount of rows, could be less).
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

datetime = "TIMESTAMP_UTC"

def process_book(book_path:str) -> str:
    """
    Process TRTH book data, that is ...
    - aggregate rows with identical timestamp (keep last)
    
    :param book_path:
        str, ...
    :return book_path_save:
        str, ...
    """
                
    # load book data
    book = pd.read_csv(book_path, parse_dates=[datetime], 
        usecols=dtype_book.keys(), dtype=dtype_book
    )
    
    # aggregate book to account for >= 2 trades having the same timestamp
    #print("aggregate book timestamps for {path}".format(
    #    path=book_path,
    #))
    #book = _aggregate_book_timestamp(book)
    
    # eliminate rounding errors to 3 decimal places
    book = book.round(3)
    
    # save book data
    book_path_save = book_path.replace(".csv", "_processed.csv")
    print("save book file as {path_save}".format(
        path_save=book_path_save,
    ))
    book.to_csv(book_path_save, index=False)
    
    return book_path_save

def process_trades(trades_path:str) -> str:
    """
    Process TRTH trades data, that is ...
    - fix timestamps
    - aggregate rows with identical timestamp (create list)

    :param trades_path:
        str, ...
    :return trades_path_save:
        str, ...
    """
    
    # load trades data
    trades = pd.read_csv(trades_path, parse_dates=[datetime],
        usecols=dtype_trades.keys(), dtype=dtype_trades,
    )
    # load corresponding book data
    book_path = trades_path.replace("Trades_", "Books_")
    book = pd.read_csv(book_path, parse_dates=[datetime], 
        usecols=dtype_book.keys(), dtype=dtype_book
    )

    # fix trades timestamp
    print("fix trades timestamp for {path}".format(
        path=trades_path,
    ))
    trades = _fix_trades_timestamp(trades, book, tolerance=200, verbose=True)
    # drop column 'Type' (needed only to identify auctions)
    trades = trades.drop(["Type"], axis=1)
    # aggregate trades to account for >= 2 trades having the same timestamp
    print("aggregate trades timestamps for {path}".format(
        path=trades_path,
    ))
    trades = _aggregate_trades_timestamp(trades)
    
    # eliminate rounding errors to 3 decimal places
    trades[["Price", "Volume"]] = (trades[["Price", "Volume"]]
        .apply(lambda row: [[round(value, 3) 
            for value in field] for field in row]
        )
    )
    
    # save trades data as .csv
    trades_path_save = trades_path.replace(".csv", "_processed.csv")
    print("save trades file as {path_save}".format(
        path_save=trades_path_save,
    ))
    trades.to_csv(trades_path_save, index=False)
    
    # save trades data as .json
    trades_path_save = trades_path.replace(".csv", "_processed.json")
    print("save trades file as {path_save}".format(
        path_save=trades_path_save,
    ))    
    data = trades.to_json(orient="records", date_format="iso")
    with open(trades_path_save, "w") as file:
        json.dump(json.loads(data), file, indent=4)
    
    return trades_path_save

if __name__ == "__main__":

    DIRECTORY = "/Volumes/HDD_8TB/EFN2/train_trades_2"
    
    for prefix, _, file_list in os.walk(DIRECTORY):
        for file_name in file_list:
            
            print("\nprocess {file_name} ...".format(
                file_name=file_name,
            ))
            file_path = os.path.join(prefix, file_name)
            
            # skip if file has already been processed or is the processed version itself
            if (
                file_path.endswith("_processed.csv") or 
                os.path.exists(file_path.replace(".csv", "_processed.csv"))
            ):
                print("already processed, continue with next file ...")
                continue
            
            # process book file
            if "Books_" in file_name:
                pass #file_path = process_book(file_path)
                
            # process trades file
            if "Trades_" in file_name:
                file_path = process_trades(file_path)                


