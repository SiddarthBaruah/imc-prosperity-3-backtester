import json
from abc import abstractmethod
from collections import deque
# from prosperity3bt.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias

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


class Strat_base_class:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def decision(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.decision(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class MarketMakingStrategy(Strat_base_class):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self._liquidity_flags = deque()
        self._flag_history_len = 10

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

        return round((most_common_bid + most_common_ask) / 2)

    def _get_highest_volume_price(self, orders: list[tuple[int, int]]) -> int:
        return max(orders, key=lambda order: abs(order[1]))[0]


class Rainforest_Resin_Strategy(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        book = state.order_depths[self.symbol]

        sell_side = book.sell_orders
        buy_side = book.buy_orders

        dominant_bid = self._extract_most_populated_price(buy_side)
        dominant_ask = self._extract_most_populated_price(sell_side)

        return round((dominant_bid + dominant_ask) / 2)

    def _extract_most_populated_price(self, side: dict[int, int]) -> int:
        return max(side.items(), key=lambda entry: abs(entry[1]))[0]

class Squid_Ink_Strategy(MarketMakingStrategy):
    def estimate_value(self, state: TradingState) -> int:
        book = state.order_depths[self.symbol]

        sell_side = book.sell_orders
        buy_side = book.buy_orders

        dominant_bid = self._extract_most_populated_price(buy_side)
        dominant_ask = self._extract_most_populated_price(sell_side)

        return round((dominant_bid + dominant_ask) / 2)

    def _extract_most_populated_price(self, side: dict[int, int]) -> int:
        return max(side.items(), key=lambda entry: abs(entry[1]))[0]


class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "KELP": KelpStrategy,
            "RAINFOREST_RESIN": Rainforest_Resin_Strategy,
            "SQUID_INK": Squid_Ink_Strategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0

        old_trader_data = json.loads(
            state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
