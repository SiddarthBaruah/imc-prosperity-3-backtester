import warnings
import json
from abc import abstractmethod
from collections import deque
# from prosperity3bt.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import pandas as pd
import numpy as np
from typing import List
from statistics import mean
import jsonpickle

JSON: TypeAlias = dict[str,
                       "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(
                        state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


PARAMS = {
    "SQUID_INK": {
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size": 0.5
    }
}


class Strat_base_class:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.conversions = 0

    @abstractmethod
    def decision(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.conversions = 0
        self.decision(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class MarketMakingStrategy(Strat_base_class):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self._liquidity_flags = deque()
        self._flag_history_len = 10

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    @abstractmethod
    def estimate_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def decision(self, state: TradingState) -> None:
        fair_price = self.estimate_value(state)

        depth = state.order_depths[self.symbol]
        asks = sorted(depth.sell_orders.items())
        bids = sorted(depth.buy_orders.items(), reverse=True)

        current_pos = state.position.get(self.symbol, 0)
        remaining_buys = self.limit - current_pos
        remaining_sells = self.limit + current_pos

        self._update_flag_window(abs(current_pos) == self.limit)
        liquidate_softly = self._should_soft_liquidate()
        liquidate_hardly = self._should_hard_liquidate()

        max_entry_bid = fair_price - 1 if current_pos > self.limit * 0.5 else fair_price
        min_entry_ask = fair_price + 1 if current_pos < self.limit * -0.5 else fair_price

        remaining_buys = self._lift_asks(asks, max_entry_bid, remaining_buys)
        remaining_buys = self._force_entry_buy(
            fair_price, remaining_buys, liquidate_hardly, liquidate_softly, bids, max_entry_bid)

        remaining_sells = self._hit_bids(bids, min_entry_ask, remaining_sells)
        remaining_sells = self._force_entry_sell(
            fair_price, remaining_sells, liquidate_hardly, liquidate_softly, asks, min_entry_ask)

    def _update_flag_window(self, full_pos: bool) -> None:
        self._liquidity_flags.append(full_pos)
        if len(self._liquidity_flags) > self._flag_history_len:
            self._liquidity_flags.popleft()

    def _should_soft_liquidate(self) -> bool:
        return (
            len(self._liquidity_flags) == self._flag_history_len and
            sum(self._liquidity_flags) >= self._flag_history_len / 2 and
            self._liquidity_flags[-1]
        )

    def _should_hard_liquidate(self) -> bool:
        return len(self._liquidity_flags) == self._flag_history_len and all(self._liquidity_flags)

    def _lift_asks(self, ask_book, price_ceiling, budget) -> int:
        for listing_price, size in ask_book:
            if budget > 0 and listing_price <= price_ceiling:
                executed = min(budget, -size)
                self.buy(listing_price, executed)
                budget -= executed
        return budget

    def _force_entry_buy(self, reference, budget, hard, soft, all_bids, price_ceiling) -> int:
        if budget > 0 and hard:
            self.buy(reference, budget // 2)
            budget -= budget // 2

        if budget > 0 and soft:
            self.buy(reference - 2, budget // 2)
            budget -= budget // 2

        if budget > 0:
            crowd_bid = max(all_bids, key=lambda x: x[1])[0]
            adjusted_price = min(price_ceiling, crowd_bid + 1)
            self.buy(adjusted_price, budget)
        return 0

    def _hit_bids(self, bid_book, price_floor, budget) -> int:
        for offer_price, size in bid_book:
            if budget > 0 and offer_price >= price_floor:
                executed = min(budget, size)
                self.sell(offer_price, executed)
                budget -= executed
        return budget

    def _force_entry_sell(self, reference, budget, hard, soft, all_asks, price_floor) -> int:
        if budget > 0 and hard:
            self.sell(reference, budget // 2)
            budget -= budget // 2

        if budget > 0 and soft:
            self.sell(reference + 2, budget // 2)
            budget -= budget // 2

        if budget > 0:
            crowd_ask = min(all_asks, key=lambda x: x[1])[0]
            adjusted_price = max(price_floor, crowd_ask - 1)
            self.sell(adjusted_price, budget)
        return 0

    def save(self) -> JSON:
        return list(self._liquidity_flags)

    def load(self, data: JSON) -> None:
        self._liquidity_flags = deque(data)


class KelpStrategy(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]

        bids = list(depth.buy_orders.items())
        asks = list(depth.sell_orders.items())

        most_common_bid = self._get_highest_volume_price(bids)
        most_common_ask = self._get_highest_volume_price(asks)

        current_mid = (most_common_bid + most_common_ask) / 2

        return round(current_mid)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class CROISSANTS(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]

        bids = list(depth.buy_orders.items())
        asks = list(depth.sell_orders.items())

        most_common_bid = self._get_highest_volume_price(bids)
        most_common_ask = self._get_highest_volume_price(asks)

        current_mid = (most_common_bid + most_common_ask) / 2

        return round(current_mid)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class Djembes_strat(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]

        bids = list(depth.buy_orders.items())
        asks = list(depth.sell_orders.items())

        most_common_bid = self._get_highest_volume_price(bids)
        most_common_ask = self._get_highest_volume_price(asks)

        current_mid = (most_common_bid + most_common_ask) / 2

        return round(current_mid)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class Jams_strat(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]

        bids = list(depth.buy_orders.items())
        asks = list(depth.sell_orders.items())

        most_common_bid = self._get_highest_volume_price(bids)
        most_common_ask = self._get_highest_volume_price(asks)

        current_mid = (most_common_bid + most_common_ask) / 2

        return round(current_mid)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class Picnic_basket_2_strat(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]

        bids = list(depth.buy_orders.items())
        asks = list(depth.sell_orders.items())

        most_common_bid = self._get_highest_volume_price(bids)
        most_common_ask = self._get_highest_volume_price(asks)

        current_mid = (most_common_bid + most_common_ask) / 2

        return round(current_mid)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class Rainforest_Resin_Strategy(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        return 10000


class Picnic_Basket1_strat(MarketMakingStrategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self._Picnic_basket1_data = None
        self.threshold = 2
        self.window_size = 500
        self.min_window_size = 100
        self.test = False
        self.position = "None"
        self.strats = {
            "none": self.do_none,
            "pairs_trade_using_syn": self.pairs_trade_strat,
            "estimate_val": self.estimate_value
        }

    def save(self):
        return jsonpickle.encode({
            "Picnic_basket1_data": self._Picnic_basket1_data,
            "Flags": list(self._liquidity_flags)
        })

    def load(self, data: JSON):
        if data is not None:
            data = jsonpickle.decode(data)
            self._Picnic_basket1_data = data.get("Picnic_basket1_data", None)
            self._liquidity_flags = deque(data.get("Flags", []))
        else:
            self._Picnic_basket1_data = None

    def do_none(self, state):
        pass

    def decision(self, state):
        # return super().decision(state)
        self.strats["none"](state)

    def pairs_trade_strat(self, state):
        if self._Picnic_basket1_data is None:
            self._Picnic_basket1_data = {
                "actual_price": [],
                "syn_price": [],
                "spread": []
            }
        croissant_value = self.get_mid_price(state, "CROISSANTS")
        djembe_value = self.get_mid_price(state, "DJEMBES")
        jam_value = self.get_mid_price(state, "JAMS")

        basket_synthetic_value = 6*croissant_value + 3*jam_value + 1*djembe_value
        basket_price = self.get_mid_price(state, self.symbol)
        current_spead = basket_price - basket_synthetic_value

        self._Picnic_basket1_data["actual_price"].append(basket_price)
        self._Picnic_basket1_data["syn_price"].append(basket_synthetic_value)
        self._Picnic_basket1_data["spread"].append(current_spead)

        # adjusting for the window
        if len(self._Picnic_basket1_data["actual_price"]) > self.window_size:
            self._Picnic_basket1_data["actual_price"].pop(0)
        if len(self._Picnic_basket1_data["syn_price"]) > self.window_size:
            self._Picnic_basket1_data["syn_price"].pop(0)
        if len(self._Picnic_basket1_data["spread"]) > self.window_size:
            self._Picnic_basket1_data["spread"].pop(0)

        if len(self._Picnic_basket1_data["actual_price"]) > self.min_window_size:
            spread_mean = np.mean(self._Picnic_basket1_data["spread"])
            spread_std = np.std(self._Picnic_basket1_data["spread"])

            z_score = (current_spead - spread_mean) / \
                spread_std if spread_std != 0 else 0
            if z_score > self.threshold:
                self.go_short(state=state)
                if self.test:
                    raise ValueError(
                        f"Short {state.timestamp} - {self.symbol} - {basket_price} - {basket_synthetic_value} - {current_spead}-{z_score}")
            elif z_score < -self.threshold:
                self.go_long(state=state)
                if self.test:
                    raise ValueError(
                        f"Long {state.timestamp} - {self.symbol} - {basket_price} - {basket_synthetic_value} - {current_spead}-{z_score}")

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, int(to_sell/2))

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, int(to_buy/2))


class JAM_DJEMBE_pairs_trade(MarketMakingStrategy):
    def __init__(self, symbol, limit):
        # super().__init__(symbol, limit)
        self.data = None
        self.threshold = 2
        self.window_size = 1000
        self.min_window_size = 200
        self.test = False
        self.position = "None"
        self.jam = "JAMS"
        self.djembe = "DJEMBES"
        self.strats = {"pairs": self.pairs_trade_strat}
        self.limit = {
            self.jam: 350,
            self.djembe: 60
        }

    def decision(self, state):
        self.strats["pairs"](state)

    def run(self, state):
        self.orders = {self.jam: [], self.djembe: []}
        self.conversions = 0
        self.decision(state)
        return self.orders
        # return super().run(state)

    def pairs_trade_strat(self, state: TradingState):
        if self.data is None:
            self.data = {
                f"{self.jam}_price": [],
                f"{self.djembe}_djembe_price": [],
                "spread": []
            }
        djembe_value = self.get_mid_price(state, self.djembe)
        jam_value = self.get_mid_price(state, self.jam)
        current_spead = djembe_value - jam_value
        self.data[f"{self.jam}_price"].append(jam_value)
        self.data[f"{self.djembe}_djembe_price"].append(djembe_value)
        self.data["spread"].append(current_spead)

        # adjusting for the window
        if len(self.data[f"{self.jam}_price"]) > self.window_size:
            self.data[f"{self.jam}_price"].pop(0)
        if len(self.data[f"{self.djembe}_djembe_price"]) > self.window_size:
            self.data[f"{self.djembe}_djembe_price"].pop(0)
        if len(self.data["spread"]) > self.window_size:
            self.data["spread"].pop(0)
        if len(self.data[f"{self.jam}_price"]) > self.min_window_size:
            spread_mean = np.mean(self.data["spread"])
            spread_std = np.std(self.data["spread"])

            z_score = (current_spead - spread_mean) / \
                spread_std if spread_std != 0 else 0
            if z_score > self.threshold:
                self.go_short(state=state, symbol=self.djembe)
                self.go_long(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Short {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")
            elif z_score < -self.threshold:
                self.go_long(state=state, symbol=self.djembe)
                self.go_short(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Long {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")

    def save(self):
        return jsonpickle.encode({
            f"{self.jam}_{self.djembe}_data": self.data,
        })

    def load(self, data: JSON):
        if data is not None:
            data = jsonpickle.decode(data)
            self.data = data.get(f"{self.jam}_{self.djembe}_data", None)
        else:
            self.data = None

    def go_short(self, state: TradingState, symbol) -> None:
        order_depth = state.order_depths[symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(symbol, 0)
        to_sell = self.limit[symbol] + position

        self.sell(price, int(to_sell/2), symbol=symbol)

    def go_long(self, state: TradingState, symbol) -> None:
        order_depth = state.order_depths[symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(symbol, 0)
        to_buy = self.limit[symbol] - position

        self.buy(price, int(to_buy/2), symbol=symbol)

    def buy(self, price: int, quantity: int, symbol) -> None:
        self.orders[symbol].append(Order(symbol, price, quantity))

    def sell(self, price: int, quantity: int, symbol) -> None:
        self.orders[symbol].append(Order(symbol, price, -quantity))


class picnic_basket_2_strat(JAM_DJEMBE_pairs_trade):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.data = None
        self.threshold = 2
        self.window_size = 1000
        self.min_window_size = 200
        self.test = False
        self.position = "None"
        self.jam = "PICNIC_BASKET1"
        # "PICNIC_BASKET1": 60,
        # "PICNIC_BASKET2": 100,
        self.djembe = "PICNIC_BASKET2"
        self.strats = {"pairs": self.pairs_trade_strat}
        self.limit = {
            self.jam: 60,
            self.djembe: 100
        }

    def pairs_trade_strat(self, state: TradingState):
        if self.data is None:
            self.data = {
                f"{self.jam}_price": [],
                f"{self.djembe}_djembe_price": [],
                "spread": []
            }
        djembe_value = self.get_mid_price(state, self.djembe)
        jam_value = self.get_mid_price(state, self.jam)
        current_spead = djembe_value - jam_value
        self.data[f"{self.jam}_price"].append(jam_value)
        self.data[f"{self.djembe}_djembe_price"].append(djembe_value)
        self.data["spread"].append(current_spead)

        # adjusting for the window
        if len(self.data[f"{self.jam}_price"]) > self.window_size:
            self.data[f"{self.jam}_price"].pop(0)
        if len(self.data[f"{self.djembe}_djembe_price"]) > self.window_size:
            self.data[f"{self.djembe}_djembe_price"].pop(0)
        if len(self.data["spread"]) > self.window_size:
            self.data["spread"].pop(0)
        if len(self.data[f"{self.jam}_price"]) > self.min_window_size:
            spread_mean = np.mean(self.data["spread"])
            spread_std = np.std(self.data["spread"])

            z_score = (current_spead - spread_mean) / \
                spread_std if spread_std != 0 else 0
            if z_score > self.threshold:
                self.go_short(state=state, symbol=self.djembe)
                # self.go_long(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Short {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")
            elif z_score < -self.threshold:
                self.go_long(state=state, symbol=self.djembe)
                # self.go_short(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Long {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")


class cross_djem_1_strat(JAM_DJEMBE_pairs_trade):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.data = None
        self.threshold = 2.5
        self.window_size = 1000
        self.min_window_size = 999
        self.test = False
        self.position = "None"
        self.jam = "PICNIC_BASKET1"
        # "PICNIC_BASKET1": 60,
        # "PICNIC_BASKET2": 100,
        self.djembe = "PICNIC_BASKET2"
        self.strats = {"pairs": self.pairs_trade_strat}
        self.limit = {
            self.jam: 60,
            self.djembe: 100
        }

    def pairs_trade_strat(self, state: TradingState):
        if self.data is None:
            self.data = {
                f"{self.jam}_price": [],
                f"{self.djembe}_djembe_price": [],
                "spread": []
            }
        djembe_value = self.get_mid_price(state, self.djembe)
        jam_value = self.get_mid_price(state, self.jam)
        current_spead = djembe_value - jam_value
        self.data[f"{self.jam}_price"].append(jam_value)
        self.data[f"{self.djembe}_djembe_price"].append(djembe_value)
        self.data["spread"].append(current_spead)

        # adjusting for the window
        if len(self.data[f"{self.jam}_price"]) > self.window_size:
            self.data[f"{self.jam}_price"].pop(0)
        if len(self.data[f"{self.djembe}_djembe_price"]) > self.window_size:
            self.data[f"{self.djembe}_djembe_price"].pop(0)
        if len(self.data["spread"]) > self.window_size:
            self.data["spread"].pop(0)
        if len(self.data[f"{self.jam}_price"]) > self.min_window_size:
            spread_mean = np.mean(self.data["spread"])
            spread_std = np.std(self.data["spread"])

            z_score = (current_spead - spread_mean) / \
                spread_std if spread_std != 0 else 0
            if z_score > self.threshold:
                self.go_short(state=state, symbol=self.djembe)
                self.go_long(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Short {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")
            elif z_score < -self.threshold:
                self.go_long(state=state, symbol=self.djembe)
                self.go_short(state=state, symbol=self.jam)
                if self.test:
                    raise ValueError(
                        f"Long {state.timestamp} - {self.symbol} - {djembe_value} - {jam_value} - {current_spead}-{z_score}")


class Squid_Ink_Strategy(MarketMakingStrategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self._Squid_Ink_data = None
        self.threshold = 3
        self.window_size = 1000
        self.min_window_size = 200
        self.test = False
        self.position = "None"

    def save(self):
        return jsonpickle.encode({
            "Squid_Ink_data": self._Squid_Ink_data,
            "Flags": list(self._liquidity_flags)
        })

    def load(self, data: JSON):
        if data is not None:
            data = jsonpickle.decode(data)
            self._Squid_Ink_data = data.get("Squid_Ink_data", None)
            self._liquidity_flags = deque(data.get("Flags", []))
        else:
            self._Squid_Ink_data = None

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def estimate_value(self, state):
        if self._Squid_Ink_data is None:
            self._Squid_Ink_data = {
                "Price_history": []
            }

        current_mid = self.get_mid_price(state, self.symbol)

        self._Squid_Ink_data["Price_history"].append(
            current_mid)

        if len(self._Squid_Ink_data["Price_history"]) > self.window_size:
            self._Squid_Ink_data["Price_history"].pop(0)

        mean_price = np.mean(self._Squid_Ink_data["Price_history"])
        std_dev = np.std(self._Squid_Ink_data["Price_history"])

        deviation = (current_mid - mean_price) / std_dev if std_dev != 0 else 0
        if deviation > self.threshold:
            if current_mid >= 2000 and len(self._Squid_Ink_data["Price_history"]) > self.min_window_size:
                self.go_short(state=state)
                # self._Squid_Ink_data["Price_history"] = []
            if self.test:
                raise ValueError(
                    f"Short {state.timestamp} - {self.symbol} - {current_mid} - {mean_price} - {std_dev}-{deviation}")
        elif deviation < -self.threshold:
            if current_mid <= 2000 and len(self._Squid_Ink_data["Price_history"]) > self.min_window_size:
                self.go_long(state=state)
                # self._Squid_Ink_data["Price_history"] = []
            if self.test:
                raise ValueError(
                    f"Long {state.timestamp} - {self.symbol} - {current_mid} - {mean_price} - {std_dev}-{deviation}")

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell//2)

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy//2)

    def decision(self, state):
        self.estimate_value(state)
        # return super().decision(state)


class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "JAMS": 350,
            "JAM_DJEMBE_pairs_trade": [350, 60],
            "PICNIC_DJEMBE_pairs_trade": [20, 30],
            "CROSS_DJEM_pairs_trade": [0, 0],
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "KELP": KelpStrategy,
            "RAINFOREST_RESIN": Rainforest_Resin_Strategy,
            "SQUID_INK": Squid_Ink_Strategy,
            "JAM_DJEMBE_pairs_trade": JAM_DJEMBE_pairs_trade,
            "PICNIC_DJEMBE_pairs_trade": picnic_basket_2_strat,
            "CROSS_DJEM_pairs_trade": cross_djem_1_strat,
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0

        old_trader_data = json.loads(
            state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, MarketMakingStrategy):
                strategy: MarketMakingStrategy

            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))
            if symbol == "JAM_DJEMBE_pairs_trade":
                orders1 = strategy.run(state)
                orders["JAMS"] = orders1["JAMS"]
                orders["DJEMBES"] = orders1["DJEMBES"]
            elif symbol == "PICNIC_DJEMBE_pairs_trade":
                orders1 = strategy.run(state)
                orders["PICNIC_BASKET2"] = orders1["PICNIC_BASKET2"]
            elif symbol == "CROSS_DJEM_pairs_trade":
                orders1 = strategy.run(state)
                orders["PICNIC_BASKET1"] = orders1["PICNIC_BASKET1"]
                # orders["DJEMBES"] = orders1["DJEMBES"]
            else:
                if symbol in state.order_depths:
                    orders[symbol] = strategy.run(state)
            conversions += strategy.conversions
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
