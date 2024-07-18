import time
import example_utils
from hyperliquid.utils import constants
from hyperliquid.utils.signing import get_timestamp_ms

order_times = []

def main():
    global order_times
    address, info, exchange = example_utils.setup(constants.TESTNET_API_URL, skip_ws=True)

    coin = "ETH"
    is_buy = False
    sz = 0.05

    print(f"We try to Market {'Buy' if is_buy else 'Sell'} {sz} {coin}.")

    order_result = exchange.market_open(coin, is_buy, sz, None, 0.01)
    if order_result["status"] == "ok":
        for status in order_result["response"]["data"]["statuses"]:
            try:
                filled = status["filled"]
                print(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                # Log the timestamp of the order arrival
                order_times.append(get_timestamp_ms())
            except KeyError:
                print(f'Error: {status["error"]}')

        print("We wait for 2s before closing")
        time.sleep(2)

        print(f"We try to Market Close all {coin}.")
        order_result = exchange.market_close(coin)
        if order_result["status"] == "ok":
            for status in order_result["response"]["data"]["statuses"]:
                try:
                    filled = status["filled"]
                    print(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                    # Log the timestamp of the order arrival
                    order_times.append(get_timestamp_ms())
                except KeyError:
                    print(f'Error: {status["error"]}')

if __name__ == "__main__":
    main()
    # After running the main function, calculate the order arrival rate
    if len(order_times) > 1:
        import numpy as np

        def calculate_order_arrival_rate(order_times: np.ndarray) -> float:
            inter_arrival_times = np.diff(order_times)
            arrival_rate = 1 / np.mean(inter_arrival_times)
            return arrival_rate

        order_times_array = np.array(order_times)
        k = calculate_order_arrival_rate(order_times_array)
        print(f"Estimated Order Arrival Rate (k): {k:.4f}")
    else:
        print("Not enough data to calculate order arrival rate.")