# This is an end to end example of a very basic adding strategy.
import json
import logging
import threading
from tabulate import tabulate
import time
from typing import Tuple, Dict, Optional, Union, Literal
import os

from examples import example_utils

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.signing import get_timestamp_ms
from hyperliquid.utils.types import (
    SIDES,
    L2BookMsg,
    L2BookSubscription,
    Literal,
    Side,
    TypedDict,
    UserEventsMsg,
)

# The maximum absolute position value the strategy can accumulate in units of the coin.
# i.e. the strategy will place orders such that it can long up to 1 ETH or short up to 1 ETH
MAX_POSITION = 100.0
MAX_ORDER_SIZE = 50.0  # Example maximum order size
REQUIRED_MARGIN = 10.0  # Example value, adjust based on your risk management strategy
COIN = "WLD"
PositionType = Literal["long", "short"]
InFlightOrder = TypedDict("InFlightOrder", {"type": Literal["in_flight_order"], "time": int})
Resting = TypedDict("Resting", {"type": Literal["resting"], "px": float, "oid": int})
Cancelled = TypedDict("Cancelled", {"type": Literal["cancelled"]})
ProvideState = Union[InFlightOrder, Resting, Cancelled]

# Dictionary to store the latest values for each metric
metrics = {
    "Bid, Ask, Mid, Spread": [0,0,0,0],
    "Optimal_price: bid, ask": [0,0],
    "Account value": 0,
    "Placing side": "",
    "Total margin used": 0,
    "Dynamic leverage": 0,
    "Available margin": 0,
    "Open orders": "",
    "Margin reserved by orders": 0,
    "Effective available margin (with leverage)": 0,
    "Buffered effective available margin": 0,
    "Max size based on margin": 0,
    "Max size based on position": 0,
    "Min order size based on $10 value": 0,
    "Calculated order size": 0,
    "Rebalancing position size": 0,
    "Rebalancing position price": 0,
    "Rebalancing position side": "",

    "Current PnL": 0,
    "Unrealized PnL": 0,
    "Closed PnL": 0
}

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_metrics():
    clear_console()
    data = [[key, value] for key, value in metrics.items()]
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))

def side_to_int(side: Side) -> int:
    return 1 if side == "A" else -1


def side_to_uint(side: Side) -> int:
    return 1 if side == "A" else 0


def calculate_mid_price_and_spread(book_data: Dict) -> Tuple[float, float]:
    # Check if the order book data is valid
    if not book_data or not book_data["levels"]:
        logging.error("Invalid order book data")
        return 0.0, 0.0

    # Extract the best bid and ask prices from the order book data
    best_bid = float(book_data["levels"][0][0]["px"])
    best_ask = float(book_data["levels"][1][0]["px"])
    

    # Calculate the mid-price and spread
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    metrics.update({
        "Bid, Ask, Mid, Spread": [best_bid, best_ask, mid_price, spread]
    })

    return mid_price, spread


def calculate_optimal_prices(mid_price: float, spread: float, volatility: float, risk_aversion: float, order_arrival_rate: float) -> Tuple[float, float]:
    half_spread = spread / 2
    adjustment = (risk_aversion * volatility**2) / (2 * order_arrival_rate)
    bid_price = mid_price - half_spread - adjustment
    ask_price = mid_price + half_spread + adjustment
    return bid_price, ask_price


def calculate_dynamic_allowable_deviation(spread: float, volatility: float) -> float:
    # Example logic: Adjust allowable deviation based on spread and volatility
    # You can customize this logic based on your requirements
    base_deviation = 0.01  # Base deviation
    spread_factor = spread / 100  # Adjust based on spread
    volatility_factor = volatility  # Adjust based on volatility
    logging.debug(f"spread_factor: {spread_factor} volatility_factor: {volatility_factor}")
    return base_deviation + spread_factor + volatility_factor

class BasicAdder:
    def __init__(self, address: str, info: Info, exchange: Exchange):
        self.info = info
        self.exchange = exchange
        subscription: L2BookSubscription = {"type": "l2Book", "coin": COIN}
        self.info.subscribe(subscription, self.on_book_update)
        self.info.subscribe({"type": "userEvents", "user": address}, self.on_user_events)
        self.position: Optional[float] = 0.0
        self.entry_price: Optional[float] = 0.0  # Track the entry price
        self.provide_state: Dict[Side, ProvideState] = {
            "A": {"type": "cancelled"},
            "B": {"type": "cancelled"},
        }
        self.recently_cancelled_oid_to_time: Dict[int, int] = {}
        self.poller = threading.Thread(target=self.poll)
        self.poller.start()
        self.rebalance_threshold = 0.05  # Example threshold for rebalancing
        self.latest_book_data: Optional[Dict] = None  # Initialize latest_book_data
        self.pnl = 0.0  # Initialize PnL
        self.closed_pnl = 0.0  # Initialize closed PnL
        self.mark_price = 0.0  # Initialize mark price
        self.stop_loss_threshold = 0.20  # Example stop-loss threshold

    def on_book_update(self, book_msg: L2BookMsg) -> None:
        logging.debug(f"book_msg {book_msg}")
        book_data = book_msg["data"]
        if book_data["coin"] != COIN:
            print("Unexpected book message, skipping")
            return

        # Store the latest order book data
        self.latest_book_data = book_data

        # Calculate mid-price and spread
        mid_price, spread = calculate_mid_price_and_spread(book_data)
        logging.debug(f"Mid-price: {mid_price}, Spread: {spread}")

        # Update the last price and mark price
        self.last_price = mid_price
        self.mark_price = mid_price

        # Calculate PnL
        self.calculate_pnl()

        # Parameters for the Avellaneda-Stoikov model
        volatility = 0.07  # Example volatility
        risk_aversion = 0.1  # Example risk aversion
        order_arrival_rate = 1.0  # Example order arrival rate

        # Calculate optimal bid and ask prices
        bid_price, ask_price = calculate_optimal_prices(mid_price, spread, volatility, risk_aversion, order_arrival_rate)
        metrics.update({
            "Optimal_price: bid, ask": [bid_price, ask_price]
        })

        # Calculate dynamic allowable deviation
        dynamic_allowable_deviation = calculate_dynamic_allowable_deviation(spread, volatility)
        logging.debug(f"Dynamic allowable deviation: {dynamic_allowable_deviation}")

        for side in SIDES:
            if side == "B":
                ideal_price = bid_price
            else:
                ideal_price = ask_price

            logging.debug(f"on_book_update ideal_price:{ideal_price}")

            # If a resting order exists, maybe cancel it
            provide_state = self.provide_state[side]
            if provide_state["type"] == "resting":
                distance = abs(ideal_price - provide_state["px"])
                if distance > dynamic_allowable_deviation * ideal_price:
                    oid = provide_state["oid"]
                    print(
                        f"cancelling order due to deviation oid:{oid} side:{side} ideal_price:{ideal_price} px:{provide_state['px']}"
                    )
                    response = self.exchange.cancel(COIN, oid)
                    if response["status"] == "ok":
                        self.recently_cancelled_oid_to_time[oid] = get_timestamp_ms()
                        self.provide_state[side] = {"type": "cancelled"}
                    else:
                        print(f"Failed to cancel order {provide_state} {side}", response)
            elif provide_state["type"] == "in_flight_order":
                if provide_state["time"] is not None and get_timestamp_ms() - provide_state["time"] > 10000:
                    print("Order is still in flight after 10s treating as cancelled", provide_state)
                    self.provide_state[side] = {"type": "cancelled"}

            # If we aren't providing, maybe place a new order
            provide_state = self.provide_state[side]
            if provide_state["type"] == "cancelled":
                position_type = "long" if side == "A" else "short"
                self.place_order(side, ideal_price, position_type)

        # Rebalance if necessary
        self.rebalance(book_data)

        # Check stop-loss
        self.stop_loss_check(self.stop_loss_threshold)

    def place_order(self, side: str, ideal_price: float, position_type: PositionType):
        if self.position is None:
            logging.debug("Not placing an order because waiting for next position refresh")
            return

        # Fetch user state and open orders from the Info API
        user_state = self.info.user_state(self.exchange.wallet.address)
        account_value = float(user_state["marginSummary"]["accountValue"])
        total_margin_used = float(user_state["marginSummary"]["totalMarginUsed"])
        open_orders = self.info.open_orders(self.exchange.wallet.address)
        
       # Debugging statement to check the content of open_orders
        print(f"Open orders: {open_orders}")

        # Check if open_orders is a list and contains the expected keys
        if isinstance(open_orders, list) and all("limitPx" in order and "sz" in order for order in open_orders):
            margin_reserved_by_orders = sum(float(order["limitPx"]) * float(order["sz"]) for order in open_orders)
        else:
            print("Open orders data structure is not as expected or is empty.")
            margin_reserved_by_orders = 0

        # Debugging statement to check the calculated margin_reserved_by_orders
        print(f"Margin reserved by orders: {margin_reserved_by_orders}")
        available_margin = account_value - total_margin_used - margin_reserved_by_orders
        
        # Calculate dynamic leverage
        LEVERAGE = self.calculate_dynamic_leverage()
        

        # Adjust available margin based on current position
        if position_type == "short":
            # Short position: account for the margin impact of the short position
            margin_impact_of_position = abs(self.position) * ideal_price / LEVERAGE
            available_margin -= margin_impact_of_position

        # Ensure available margin is positive
        if available_margin <= 0:
            print(f"Available margin is negative or zero: {available_margin}")
            return

        # Adjust available margin based on leverage
        effective_available_margin = max(available_margin * LEVERAGE, 0)  # Ensure it's not negative

        # Maintain a buffer margin to avoid consuming all available margin
        BUFFER_MARGIN = 100.0  # Increased buffer margin
        buffered_effective_available_margin = max(effective_available_margin - BUFFER_MARGIN, 0)

        max_size_based_on_margin = buffered_effective_available_margin / ideal_price
        max_size_based_on_position = MAX_POSITION - abs(self.position)
        sz = min(max_size_based_on_position, max_size_based_on_margin)

        # Ensure the order size is at least $10 in value
        MIN_ORDER_VALUE = 10.0  # Minimum order value in USD
        min_order_size = MIN_ORDER_VALUE / ideal_price

        # Update metrics
        metrics.update({
            "Placing side": position_type,
            "Dynamic leverage": LEVERAGE,
            "Calculated order size": sz,
            "Available margin": available_margin,
            "Open orders": open_orders,
            "Margin reserved by orders": margin_reserved_by_orders,
            "Effective available margin (with leverage)": effective_available_margin,
            "Buffered effective available margin": buffered_effective_available_margin,
            "Max size based on margin": max_size_based_on_margin,
            "Max size based on position": max_size_based_on_position,
            "Min order size based on $10 value": min_order_size,
            "Rebalancing position size": sz,
            "Rebalancing position price": ideal_price,
            "Rebalancing position side": side,
            "Account value": account_value,
            "Total margin used": total_margin_used
        })
        print_metrics()

        if abs(sz) < min_order_size:
            print(f"Order size {sz} is below the minimum order size {min_order_size} based on $10 value")
            return

        # Round the order size to the nearest valid increment
        MIN_ORDER_SIZE = 10  # Example minimum order size, adjust as needed
        MAX_ORDER_SIZE = 100  # Example maximum order size, adjust as needed
        ORDER_SIZE_INCREMENT = 1  # Example increment, adjust as needed
        sz = round(sz / ORDER_SIZE_INCREMENT) * ORDER_SIZE_INCREMENT

        if abs(sz) < MIN_ORDER_SIZE or abs(sz) > MAX_ORDER_SIZE:
            print(f"Order size {sz} is outside the acceptable range ({MIN_ORDER_SIZE} - {MAX_ORDER_SIZE})")
            return

        # Adjust the order price to avoid immediate matching
        if side == "B":
            ideal_price -= 0.002  # Adjust bid price slightly lower to avoid immediate matching
        else:
            ideal_price += 0.002  # Adjust ask price slightly higher to avoid immediate matching

        px = float(f"{ideal_price:.5g}")  # prices should have at most 5 significant digits
        print(f"placing order sz:{sz} px:{px} side:{side}")
        self.provide_state[side] = {"type": "in_flight_order", "time": get_timestamp_ms()}
        response = self.exchange.order(COIN, side == "B", sz, px, {"limit": {"tif": "Alo"}})
        print("placed order", response)
        if response["status"] == "ok":
            status = response["response"]["data"]["statuses"][0]
            if "resting" in status:
                self.provide_state[side] = {"type": "resting", "px": px, "oid": status["resting"]["oid"]}
                # Update entry price when order is placed
                self.entry_price = px
            else:
                print("Unexpected response from placing order. Setting position to None.", response)
                self.provide_state[side] = {"type": "cancelled"}
                self.position = None
        else:
            print("Order placement failed", response)
            self.provide_state[side] = {"type": "cancelled"}

        # Print the updated metrics
        print_metrics()

    def rebalance(self, book_data: Dict):
        if self.position is not None and abs(self.position) > self.rebalance_threshold * MAX_POSITION:
            # Use the latest order book data
            mid_price, spread = calculate_mid_price_and_spread(book_data)
            volatility = 0.07  # Example volatility
            risk_aversion = 0.1  # Example risk aversion
            order_arrival_rate = 1.0  # Example order arrival rate
            bid_price, ask_price = calculate_optimal_prices(mid_price, spread, volatility, risk_aversion, order_arrival_rate)

            side = "A" if self.position > 0 else "B"
            position_type = "long" if self.position > 0 else "short"
            ideal_price = ask_price if side == "A" else bid_price
            sz = min(abs(self.position), MAX_ORDER_SIZE)  # Adjust order size dynamically
            px = float(f"{ideal_price:.5g}")
            print(f"Rebalancing position sz:{sz} px:{px} side:{side}")

            if self.check_margin():
                self.place_order(side, px, position_type)

            # Update metrics
            metrics.update({
                "Rebalancing position size": sz,
                "Rebalancing position price": ideal_price,
                "Rebalancing position side": side
            })
            print_metrics()

    def calculate_dynamic_leverage(self) -> float:
        # Example logic to calculate dynamic leverage
        # You can customize this logic based on your risk management strategy
        base_leverage = 5  # Base leverage value
        market_conditions_factor = 1.0  # Adjust based on market conditions
        return base_leverage * market_conditions_factor

    def stop_loss_check(self, stop_loss_threshold: float):
        # Ensure position and current_pnl are not None before checking stop-loss
        if self.position is not None and self.current_pnl is not None:
            if self.position < 0 and self.current_pnl < -stop_loss_threshold:
                self.close_position()
            elif self.position > 0 and self.current_pnl < -stop_loss_threshold:
                self.close_position()

    def close_position(self):
        # Logic to close the current position
        if self.position < 0:
            self.place_order("B", self.mark_price, "short")
        elif self.position > 0:
            self.place_order("A", self.mark_price, "long")

    def calculate_pnl(self):
        # Calculate the unrealized PnL based on the current position and the last price
        if self.position is not None and self.entry_price is not None:
            if self.position > 0:
                # Long position
                unrealized_pnl = (self.mark_price - self.entry_price) * self.position
            else:
                # Short position
                unrealized_pnl = (self.entry_price - self.mark_price) * abs(self.position)
        else:
            unrealized_pnl = 0.0

        total_pnl = self.closed_pnl + unrealized_pnl

        # Update metrics
        metrics.update({
            "Current PnL": total_pnl,
            "Unrealized PnL": unrealized_pnl,
            "Closed PnL": self.closed_pnl
        })
        print_metrics()
        self.current_pnl = total_pnl  # Update current PnL

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        print(user_events)
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            for fill in user_events_data["fills"]:
                fill_px = float(fill["px"])
                fill_sz = float(fill["sz"])
                fill_side = fill["side"]
                fill_pnl = float(fill["closedPnl"])
                if fill_side == "B":
                    if self.position is None:
                        self.position = 0.0
                    self.position += fill_sz
                    if self.entry_price is None:
                        self.entry_price = fill_px
                    else:
                        self.entry_price = (self.entry_price * (self.position - fill_sz) + fill_px * fill_sz) / self.position
                else:
                    if self.position is None:
                        self.position = 0.0
                    self.position -= fill_sz
                self.closed_pnl += fill_pnl  # Accumulate closed PnL
                self.pnl += fill_pnl

        # Check stop-loss
        self.stop_loss_check(self.stop_loss_threshold)

    def check_margin(self) -> bool:
        # Fetch user state and open orders from the Info API
        user_state = self.info.user_state(self.exchange.wallet.address)
        account_value = float(user_state["marginSummary"]["accountValue"])
        #print(f"account_value: {account_value}")
        total_margin_used = float(user_state["marginSummary"]["totalMarginUsed"])
        #print(f"total_margin_used: {total_margin_used}")

        # Calculate the margin reserved by open orders
        open_orders = self.info.open_orders(self.exchange.wallet.address)
        margin_reserved_by_orders = sum(float(order["limitPx"]) * float(order["sz"]) for order in open_orders)

        # Calculate available margin
        available_margin = account_value - total_margin_used - margin_reserved_by_orders
        #print(f"available_margin: {available_margin}")
        return available_margin > REQUIRED_MARGIN

    def poll(self):
        while True:
            open_orders = self.info.open_orders(self.exchange.wallet.address)
            #print("open_orders", open_orders)
            ok_oids = set(self.recently_cancelled_oid_to_time.keys())
            for provide_state in self.provide_state.values():
                if provide_state["type"] == "resting":
                    ok_oids.add(provide_state["oid"])
            
            for open_order in open_orders:
                if open_order["coin"] == COIN and open_order["oid"] not in ok_oids:
                    print("Cancelling unknown oid", open_order["oid"])
                    self.exchange.cancel(open_order["coin"], open_order["oid"])

            current_time = get_timestamp_ms()
            self.recently_cancelled_oid_to_time = {
                oid: timestamp
                for (oid, timestamp) in self.recently_cancelled_oid_to_time.items()
                if current_time - timestamp > 30000
            }

            user_state = self.info.user_state(self.exchange.wallet.address)
            for position in user_state["assetPositions"]:
                if position["position"]["coin"] == COIN:
                    self.position = float(position["position"]["szi"])
                    print(f"set position to {self.position}")
                    break
            if self.position is None:
                self.position = 0.0

            # Check available margin and cancel orders if necessary
            account_value = float(user_state["marginSummary"]["accountValue"])
            total_margin_used = float(user_state["marginSummary"]["totalMarginUsed"])
            margin_reserved_by_orders = sum(float(order["limitPx"]) * float(order["sz"]) for order in open_orders)
            available_margin = account_value - total_margin_used - margin_reserved_by_orders
            print(f"available_margin: {available_margin}")

            if available_margin <= 0:
                for open_order in open_orders:
                    if open_order["coin"] == COIN:
                        print(f"Cancelling order to free up margin: oid {open_order['oid']}")
                        self.exchange.cancel(open_order["coin"], open_order["oid"])

            time.sleep(10)

            # Check stop-loss
            self.stop_loss_check(self.stop_loss_threshold)

def main():
    # Setting this to logging.DEBUG can be helpful for debugging websocket callback issues
    logging.basicConfig(level=logging.ERROR)
    address, info, exchange = example_utils.setup(constants.TESTNET_API_URL)
    BasicAdder(address, info, exchange)

if __name__ == "__main__":
    main()