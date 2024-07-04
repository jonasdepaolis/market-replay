# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

"""
Given below are the columns and dtypes corresponding to the different source
types: 'BOOK'.

- 'BOOK' is based on TRTH book data, including a timestamp (UTC) and 10 limit
order book levels, each with bid/ask price/size.
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

def _aggregate_book_timestamp(
        book:pd.DataFrame
    ):
    """
    Handle non-unique timestamp-based index. Note that this step is necessary
    so the sources can be properly aligned!

    :param book:
        pd.DataFrame, book for a given stock
    :return book:
        pd.DataFrame, book for a given stock with aggregated rows
    """

    datetime = "TIMESTAMP_UTC"

    # aggregate multiple rows using a single timestamp (keep last duplicate)
    book = book.drop_duplicates(
        subset=datetime, keep="last", inplace=False
    )
    
    # eliminate rounding errors to 3 decimal places
    book = book.round(3)

    return book

if __name__ == "__main__":

    # load trades data
    book = pd.read_csv(
        "/Volumes/HDD_8TB/_READ_ONLY/TRTH_BOOK/Books_ADSGn.DE_20160101_20160131.csv",
        usecols=dtype_book.keys(), dtype=dtype_book, parse_dates=["TIMESTAMP_UTC"]
    )
    # aggregate trades data
    book_aggregated = _aggregate_book_timestamp(book)


