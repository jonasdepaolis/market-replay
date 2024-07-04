# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# use relative imports for other modules 
from env.market import MarketState, Order, Trade

# general imports
import copy
import datetime
import logging
logging.basicConfig(level=logging.CRITICAL) # logging.basicConfig(level=logging.NOTSET)
import os
import pandas as pd
import random
import time

SOURCE_DIRECTORY = "/home/jovyan/_shared_storage/read_only/efn2_backtesting"
DATETIME = "TIMESTAMP_UTC"

class Episode:

    def __init__(self,
        identifier_list:list,
        episode_start_buffer:str,
        episode_start:str,
        episode_end:str,
    ):
        """
        Prepare a single episode as a generator. The episode is the main 
        building block of each backtest. 

        :param identifier_list:
            pd.Timestamp, start building the market state, ignore agent

        :param episode_start_buffer:
            str, timestamp from which to start building the market state, ignore agent
        :param episode_start:
            str, timestamp from which to start informing the agent
        :param episode_end:
            str, timestamp from which to stop informing the agent
        """

        # data settings
        self.identifier_list = identifier_list

        # ...
        self._episode_start_buffer = pd.Timestamp(episode_start_buffer)
        self._episode_start = pd.Timestamp(episode_start)
        self._episode_end = pd.Timestamp(episode_end)

        # setup routine
        self._episode_setup(
            max_deviation_tol=300, # in seconds
        ) 

        # dynamically set attributes (on per-update basis)
        self._episode_buffering = None

    # static attributes ---

    @property
    def episode_start_buffer(self): 
        return self._episode_start_buffer 
    
    @property
    def episode_start(self):
        return self._episode_start 

    @property
    def episode_end(self):
        return self._episode_end

    # dynamic attributes ---

    @property
    def episode_buffering(self):
        return self._episode_buffering

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def timestamp_next(self):
        return self._timestamp_next

    # episode setup ---

    # DONE
    def _episode_setup(self, max_deviation_tol=300):
        """
        Load and prepare episode for the  

        :param max_deviation_tol:
            int, maximum deviation from the expected episode length (in seconds)
        """

        # display progress ---

        # info
        logging.info("(INFO) episode from {episode_start} ({episode_start_buffer}) to {episode_end} is being prepared ...".format(
            episode_start=self._episode_start,
            episode_start_buffer=self._episode_start_buffer.time(),
            episode_end=self._episode_end,
        ))

        # in the beginning, unset flag to disallow iteration via __iter__ method
        self._episode_available = False

        # prepare data ---

        # build path_store to host all paths to load data from (only for this particular episode)
        path_store = self._build_path_store(self._episode_start, self._episode_end)
        # build data_store to host all data (only for this particular episode)
        data_store = self._build_data_store(self._episode_start, self._episode_end, path_store)
        # align data_store so that each data source has equal length
        data_store = self._align_data_store(data_store)
        # build data_monitor to iterate over
        data_monitor = self._build_data_monitor(data_store)

        # set attributes ---

        # set data_store to iterate over using the __iter__ method
        self._data_store = data_store
        # set data_monitor to iterate over using the __iter__ method
        self._data_monitor = data_monitor

        # sanity check ---

        # total time_delta should not deviate from episode_length by more than <tolerance> seconds
        time_delta_observed = (
            abs(self._data_monitor.iloc[0, 0] - self._episode_start) +
            abs(self._data_monitor.iloc[-1, 0] - self._episode_end)
        )
        # ...
        time_delta_required = pd.Timedelta(max_deviation_tol, "s")
       
        # ... 
        assert time_delta_observed < time_delta_required, \
            "(ERROR) time delta exceeded max deviation tolerance (required: {required}, observed: {observed})".format(
                required=time_delta_required, 
                observed=time_delta_observed, 
            )
        
        # allow iteration ---

        # set flag to allow iteration via __iter__method
        self._episode_available = True

        # info
        logging.info("(INFO) episode has successfully been set and includes a total of {num_steps} steps".format(
            num_steps=len(data_monitor.index),
        ))

    # helper methods ---

    def _build_path_store(self, timestamp_start, timestamp_end):
        """
        Find paths and store them in the path_store dictionary together with 
        their corresponding key.
        
        Note that timestamp_start and timestamp_end must belong to the same 
        date!

        :param timestamp_start:
            pd.Timestamp, ...
        :param timestamp_end:
            pd.Timestamp, ...

        :return path_store:
            dict, {<identifier>: <path>, *}
        """

        path_store = dict()

        # ...
        assert timestamp_start.date() == timestamp_end.date(), \
            "(ERROR) timestamp_start and timestamp_end must belong to the same date"
        date = timestamp_start.date()
        date_string = str(date).replace("-", "")

        # path_list includes all paths available in directory
        path_list = [os.path.join(pre, file) for pre, _, sub in os.walk(SOURCE_DIRECTORY)
            for file in sub if not file.startswith((".", "_"))
        ]

        # ...
        for identifier in self.identifier_list:

            # identify matching criteria
            market_id, event_id = identifier.split(".")

            # make copy of path_list
            path_list_ = path_list.copy()
            # filter based on matching criteria
            path_list_ = filter(
                lambda path: market_id.lower() in path.lower(), path_list_)
            path_list_ = filter(
                lambda path: event_id.lower() in path.lower(), path_list_)
            path_list_ = filter(
                lambda path: date_string in path, path_list_)
            # ...
            path_list_ = list(path_list_)

            # if path_list_this is empty, raise Exception that is caught in calling method
            if not len(path_list_) == 1:
                raise Exception("(ERROR) could not find path for {identifier} between {timestamp_start} and {timestamp_end}".format(
                    identifier=identifier,
                    timestamp_start=timestamp_start, timestamp_end=timestamp_end,
                ))

            # there should be exactly one matching path
            path = path_list_[0]

            # add dataframe to output dictionary
            path_store[identifier] = path

        # info
        logging.info("(INFO) path_store has been built")

        return path_store

    def _build_data_store(self, timestamp_start, timestamp_end, path_store):
        """
        Load .csv(.gz) and .json files into dataframes and store them in the
        data_store dictionary together with their corresponding key.

        :param path_store:
            dict, {<identifier>: <path>, *}
        :param timestamp_start:
            pd.Timestamp, ...
        :param timestamp_start:
            pd.Timestamp, ...

        :return data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, original timestamps
        """

        data_store = dict()

        # ...
        for identifier in self.identifier_list:

            # load event_id 'BOOK' as .csv(.gz)
            if "BOOK" in identifier:
                df = pd.read_csv(path_store[identifier], parse_dates=[DATETIME])
            # load event_id 'TRADES' as .json
            if "TRADES" in identifier:
                df = pd.read_json(path_store[identifier], convert_dates=True)

            # if dataframe is empty, raise Exception that is caught in calling method
            if not len(df.index) > 0:
                raise Exception("(ERROR) could not find data for {identifier} between {timestamp_start} and {timestamp_end}".format(
                    identifier=identifier,
                    timestamp_start=timestamp_start, timestamp_end=timestamp_end,
                ))

            # make timestamp timezone-unaware
            df[DATETIME] = pd.DatetimeIndex(df[DATETIME]).tz_localize(None)
            # filter dataframe to include only rows with timestamp between timestamp_start and timestamp_end
            df = df[df[DATETIME].between(timestamp_start, timestamp_end)]

            # add dataframe to output dictionary
            data_store[identifier] = df

        # info
        logging.info("(INFO) data_store has been built")

        return data_store

    def _align_data_store(self, data_store):
        """
        Consolidate and split again all sources so that each source dataframe
        contains a state for each ocurring timestamp across all sources.

        :param data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, original timestamps

        :return data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, aligned timestamps
        """

        # unpack dictionary
        id_list, df_list = zip(*data_store.items())

        # rename columns and use id as prefix, exclude timestamp
        add_prefix = lambda id, df: df.rename(columns={x: f"{id}__{x}"
            for x in df.columns[1:]
        })
        df_list = list(map(add_prefix, id_list, df_list))

        # join df_list into df_merged (full outer join)
        df_merged = pd.concat([
            df.set_index(DATETIME) for df in df_list
        ], axis=1, join="outer").reset_index()

        # split df_merged into original df_list (all df are of equal length)
        df_list = [pd.concat([
            df_merged[[DATETIME]], # global timestamp
            df_merged[[x for x in df_merged.columns if id in x] # filtered by identifier
        ]], axis=1) for id in id_list]

        # rename columns and remove prefix, exclude timestamp
        del_prefix = lambda df: df.rename(columns={x: x.split("__")[1]
            for x in df.columns[1:]
        })
        df_list = list(map(del_prefix, df_list))

        # pack dictionary
        data_store = dict(zip(id_list, df_list))

        # info
        logging.info("(INFO) data_store has been aligned")

        return data_store

    def _build_data_monitor(self, data_store):
        """
        In addition to the sources dict, return a monitor dataframe that keeps
        track of changes in state across all sources.

        :param data_store:
            dict, {<source_id>: <pd.DataFrame>, *}, aligned timestamp

        :return data_monitor:
            pd.DataFrame, changes per source and timestamp
        """

        # setup dictionary based on timestamp
        datetime_index = list(data_store.values())[0][DATETIME]
        data_monitor = {DATETIME: datetime_index}

        # track changes per source and timestamp in series
        for key, df in data_store.items():
            data_monitor[key] = ~ df.iloc[:, 1:].isna().all(axis=1)


        # build monitor as dataframe from series
        data_monitor = pd.DataFrame(data_monitor)

        # info
        logging.info("(INFO) data_monitor has been built")

        return data_monitor

    # iteration ---
        
    def __next__(self):
        pass

    def __iter__(self):
        """
        Iterate over the set episode. 
        
        NOTE: Use the self._episode_buffering flag to check if the buffering 
        phase has ended - only then should the agent be notified about market 
        updates. 
        """
        
        # return, that is, disallow iteration if no episode has been set
        if not self._episode_available:
            return
        
        # ...
        logging.info("(INFO) episode has started ...")
        
        # set buffer flag
        self._episode_buffering = True

        # time
        time_start = time.time()

        # ...
        for step, timestamp, *monitor_state in self._data_monitor.itertuples():

            # update timestamps ---

            # track this timestamp
            self._timestamp = self._data_monitor.iloc[step, 0]
            
            # track next timestamp, prevent IndexError that would arise with the last step
            self._timestamp_next = self._data_monitor.iloc[min(
                step + 1, len(self._data_monitor.index) - 1
            ), 0]

            # display progress ---

            # ...
            progress = timestamp.value / (self._episode_end.value - self._episode_start_buffer.value)
            eta = (time.time() - time_start) / progress

            # info
            logging.info("(INFO) step {step}, progress {progress}, eta {eta}".format(
                step=step,
                progress=progress,
                eta=eta,
            ))

            # handle buffer phase ---
            
            # update buffer flag, agent should start being informed only after buffering phase has ended
            cache_episode_buffering = self._episode_buffering
            self._episode_buffering = timestamp < self._episode_start
            
            # info
            if cache_episode_buffering != self._episode_buffering:
                logging.info("(INFO) buffering phase for this episode has ended, allow trading ...")
            
            # find data ---

            # get identifier (column name) per updated source (based on self._data_monitor)
            identifier_list = (self._data_monitor
                .iloc[:, 1:]
                .columns[monitor_state]
                .values
            )

            # get data per updated source (based on self._data_store)
            data_list = [self._data_store[identifier].iloc[step, :] 
                for identifier in identifier_list
            ]

            # yield data ---

            # for each step, yield update via dictionary
            update = dict(zip(identifier_list, data_list)) # {<identifier>: <data>, *}

            # ...
            yield update
        
        # time
        time_end = time.time()
        
        # ...
        time_delta = round(time_end - time_start, 3)
        time_per_step = round((time_end - time_start) / step, 3)

        # info
        logging.info("(INFO)... episode has ended, took {time_delta}s for {step} steps ({time_per_step}s/step)".format(
            time_delta=time_delta,
            step=step,
            time_per_step=time_per_step,
        ))

class Backtest:

    timestamp_global = None

    def __init__(self,
        agent, # backtest is wrapper for trading agent
    ):
        """
        Backtest wrapper that is used to evaluate a trading agent on one or 
        multiple episodes of historical market data. 

        Note that the original _agent is used only to derive a fresh copy with 
        each additional episode. 

        :param agent:
            Agent, trading agent instance that is to be evaluated
        """

        # from arguments
        self._agent = agent 

        # TODO: ...
        self.result_list = []    

    # market/agent step ---

    def _market_step(self, market_id, book_update, trade_update):
        """
        Update post-trade market state and match standing orders against 
        pre-trade market state.

        :param market_id:
            str, market identifier
        :param book_update:
            pd.Series, ...
        :param trade_update:
            pd.Series, ...
        """

        # update market state
        MarketState.instances[market_id].update(
            book_update=book_update,
            trade_update=trade_update,
        )

        # match standing agent orders against pre-trade state
        MarketState.instances[market_id].match()

    def _agent_step(self, source_id, either_update, timestamp, timestamp_next):
        """
        Inform trading agent about either book or trades state through the 
        corresponding method. Also, inform trading agent about this and next 
        timestamp. 

        :param source_id:
            str, source identifier
        :param either_update:
            pd.Series, ...
        :param timestamp:
            pd.Timestamp, ...
        :param timestamp_next:
            pd.Timestamp, ...
        """

        # case 1: alert agent every time that book is updated
        if source_id.endswith("BOOK"):
            self.agent.on_quote(market_id=source_id.split(".")[0], 
                book_state=either_update,
            )
        # case 2: alert agent every time that trade happens
        elif source_id.endswith("TRADES"):
            self.agent.on_trade(market_id=source_id.split(".")[0],
                trades_state=either_update,
            )
        # unknown source_id
        else:
            raise Exception("(ERROR) unable to parse source_id '{source_id}'".format(
                source_id=source_id, 
            ))
        
        # _always_ alert agent with time interval between this and next timestamp
        self.agent.on_time(
            timestamp=timestamp,
            timestamp_next=timestamp_next,
        )

    # option 1: run single episode ---

    def run(self, 
        identifier_list:list,
        episode_start_buffer:str,
        episode_start:str,
        episode_end:str,
        display_interval:int=100,
    ):  
        """
        Run agent against a single backtest instance based on a specified 
        episode. 

        :param identifier_list:
            list, <market_id>.BOOK/TRADES identifier for each respective data source
        :param episode_start_buffer:
            pd.Timestamp, 
        :param episode_start:
            pd.Timestamp, ...
        :param episode_end:
            pd.Timestamp, ...
        """

        # build episode ---

        # try to build episode based on the specified parameters
        try:
            episode = Episode(
                identifier_list=identifier_list,
                episode_start_buffer=episode_start_buffer,
                episode_start=episode_start,
                episode_end=episode_end,
            )
        # return if episode could not be generated
        except:
            logging.info("(ERROR) could not run episode with the specified parameters")
            return # do nothing

        # setup agent ---

        # create fresh copy of the original agent instance
        self.agent = copy.copy(self._agent)

        # setup market environment ---

        # identify market instances based on market_id
        identifier_list = set(identifier.split(".")[0] for identifier
            in identifier_list
        )
        # create market_state instances
        for market_id in identifier_list:
            _ = MarketState(market_id)

        # iterate over episode ---

        # ...
        for step, update_store in enumerate(episode, start=1): 
            
            # update global timestamp
            self.__class__.timestamp_global = episode.timestamp

            # ...
            market_list = set(identifier.split(".")[0] for identifier in update_store)
            source_list = list(update_store)

            # step 1: update book_state -> based on original data
            # step 2: match standing orders -> based on pre-trade state
            for market_id in market_list:
                self._market_step(market_id=market_id, 
                    book_update=update_store.get(f"{market_id}.BOOK"),
                    trade_update=update_store.get(f"{market_id}.TRADES", pd.Series([None] * 3)), # optional, default to empty pd.Series
                )

            # during the buffer phase, do not inform agent about update
            if episode.episode_buffering:
                continue

            # step 3: inform agent -> based on original data
            for source_id in source_list: 
                self._agent_step(source_id=source_id, 
                    either_update=update_store.get(source_id),
                    timestamp=episode.timestamp,
                    timestamp_next=episode.timestamp_next,
                )

            # finally, report the current state of the agent
            if not (step % display_interval):
                print(self.agent)
        
        # report result ---

        # TODO: ...
        result = {
            'Orders': self.agent.market_interface.order_list.copy(),
            'Trades': self.agent.market_interface.trade_list.copy(),
            'Exposure': self.agent.market_interface.exposure.copy(),
            'PnL_realized': self.agent.market_interface.pnl_realized.copy(),
            'PnL_unrealized': self.agent.market_interface.pnl_unrealized.copy(),
        }

        # save report
        self.result_list.append(result)

        # reset agent ---

        # ...
        del self.agent

        # reset market environment ---

        # delete all MarketState instances in MarketState.instances class attribute
        MarketState.reset_instances()
        # delete all Order instances in Order.history class attribute
        Order.reset_history()
        # delete all Trade instances in Trade.history class attribute
        Trade.reset_history()

        return True  # return successful episode

    # option 2: run multiple episodes ---

    def run_episode_generator(self, 
        identifier_list:list,
        date_start:str="2016-01-01",
        date_end:str="2016-03-31",
        episode_interval:int=30, # timestamp quantization
        episode_shuffle:bool=True,
        episode_buffer:int=5,
        episode_length:int=30, 
        num_episodes:int=10,
        seed=None,
    ):
        """
        Run agent against a series of generated episodes, that is, run a similar 
        episode (in terms of episode_buffer and episode_length) multiple times. 
        Call Backtest.run(...) under the hood. 

        :param identifier_list:
            list, <market_id>.BOOK/TRADES identifier for each respective data source
        :param date_start:
            str, start date after which episodes are generated, default is "2016-01-01"
        :param date_end:
            str, end date before which episodes are generated, default is "2016-03-31"
        :param episode_interval:
            int, ...
        :param episode_shuffle:
            bool, ...
        :param episode_buffer:
            int, ...
        :param episode_length:
            int, ...
        :param num_episodes:
            int, ...
        :param seed:
            None or int, if specified seed is set for generating random numbers
        """

        # pd.Timestamp 
        date_start = pd.Timestamp(date_start)
        date_end = pd.Timestamp(date_end)
        
        # pd.Timedelta
        episode_buffer = pd.Timedelta(episode_buffer, "min")
        episode_length = pd.Timedelta(episode_length, "min")

        # build episode_start_list ---

        # ...
        episode_start_list = pd.date_range(start=date_start, end=date_end + pd.Timedelta("1d"),
            freq=f"{episode_interval}min",
            normalize=True, # start at 00:00:00.000
        )

        # ...
        test_list = [
            lambda timestamp: timestamp.weekday() not in [5, 6], # sat, sun
            lambda timestamp: datetime.time(8, 0, 0) <= timestamp.time(), # valid start
            lambda timestamp: (timestamp + episode_length).time() <= datetime.time(16, 30, 0), # valid end
            # ...
        ]
        episode_start_list = [start for start in episode_start_list 
            if all(test(start) for test in test_list)
        ]

        # ...
        if seed:
            random.seed(seed)

        if episode_shuffle:
            random.shuffle(episode_start_list)

        # iterate over episode_start_list ---

        # ...
        episode_counter = 0
        episode_index = 0

        # take next episode until ...
        while episode_counter < min(len(episode_start_list), num_episodes):
            
            # ...
            status = self.run(identifier_list=identifier_list,
                episode_start_buffer=episode_start_list[episode_index],
                episode_start=episode_start_list[episode_index] + pd.Timedelta(episode_buffer, "min"),
                episode_end=episode_start_list[episode_index] + pd.Timedelta(episode_length, "min"),
            )

            # in either case, update index
            episode_index = episode_index + 1

            # update counter only if episode has been successfully run
            if status:
                episode_counter = episode_counter + 1

    def run_episode_broadcast(self, 
        identifier_list:list,
        date_start:str="2016-01-01", 
        date_end:str="2016-03-31", 
        time_start_buffer:str="08:00:00",
        time_start:str="08:10:00", 
        time_end:str="16:30:00", 
    ):
        """
        Run agent against a series of broadcast episodes, that is, run the same 
        episode (in terms of time_start_buffer, time_start, and time_end) for 
        each date between date_start and date_end. Uses Backtest.run(...) under 
        the hood. 

        :param identifier_list:
            list, <market_id>.BOOK/TRADES identifier for each respective data source
        :param date_start:
            str, start date after which episodes are generated, default is "2016-01-01"
        :param date_end:
            str, end date before which episodes are generated, default is "2016-03-31"
        :param time_start_buffer:
            int, ...
        :param time_start:
            bool, ...
        :param time_end:
            int, ...
        """

        # pd.Timestamp
        date_start = pd.Timestamp(date_start)
        date_end = pd.Timestamp(date_end)

        # pd.Timedelta
        time_start_buffer = pd.Timedelta(time_start_buffer)
        time_start = pd.Timedelta(time_start)
        time_end = pd.Timedelta(time_end)

        # build episode_date_list ---

        # ...
        episode_date_list = pd.date_range(start=date_start, end=date_end + pd.Timedelta("1d"),
            freq="1d", 
            normalize=True, # start at 00:00:00.000
        )

        # ...
        test_list = [
            lambda timestamp: timestamp.weekday() not in [5, 6], # sat, sun
            # ...
        ]
        episode_date_list = [timestamp_start for timestamp_start in episode_date_list 
            if all(test(timestamp_start) for test in test_list)
        ]

        # iterate over episode_date_list ---

        # for each date + broadcast time_start_buffer, time_start, and time_end ...
        for episode_date in episode_date_list:

            # ...
            self.run(identifier_list=identifier_list,
                episode_start_buffer=episode_date + time_start_buffer,
                episode_start=episode_date + time_start,
                episode_end=episode_date + time_end,
            )

    def run_episode_list(self, 
        identifier_list:list,
        episode_list:list, 
    ):
        """
        Run agent against a series of specified episodes, that is, work through 
        the episode_list. Uses Backtest.run(...) under the hood. 

        :param identifier_list:
            list, <market_id>.BOOK/TRADES identifier for each respective data source
        :param episode_list:
            list, includes (episode_start_buffer, episode_start, episode_end) tuples
        """

        # iterate over episode_list ---

        # for each episode ...
        for episode_start_buffer, episode_start, episode_end in episode_list:

            # ...
            self.run(identifier_list=identifier_list,
                episode_start_buffer=pd.Timestamp(episode_start_buffer), 
                episode_start=pd.Timestamp(episode_start), 
                episode_end=pd.Timestamp(episode_end), 
            )


