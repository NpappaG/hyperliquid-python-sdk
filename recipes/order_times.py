import time
import numpy as np
from hyperliquid.utils.signing import get_timestamp_ms

order_times = []

def generate_example_order_times():
    global order_times
    # Simulate order timestamps (e.g., every 2 seconds)
    for _ in range(10):  # Generate 10 example timestamps
        order_times.append(get_timestamp_ms())
        time.sleep(2)  # Simulate a 2-second interval between orders

def calculate_order_arrival_rate(order_times: np.ndarray) -> float:
    inter_arrival_times = np.diff(order_times)
    arrival_rate = 1 / np.mean(inter_arrival_times)
    return arrival_rate

def main():
    generate_example_order_times()

    if len(order_times) > 1:
        order_times_array = np.array(order_times)
        k = calculate_order_arrival_rate(order_times_array)
        print(f"Estimated Order Arrival Rate (k): {k:.4f}")
    else:
        print("Not enough data to calculate order arrival rate.")

if __name__ == "__main__":
    main()