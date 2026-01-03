#!/usr/bin/env python3
"""
Test script for order management.
Tests order placement, tracking, and position management with paper trading.

IMPORTANT: This uses your paper trading account. Orders will be real (paper) orders.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.connection import ConnectionManager
from src.orders import OrderManager, PositionManager, OrderAction
from src.data_fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_orders():
    """Test order management functionality."""

    print("=" * 60)
    print("ORDER MANAGEMENT TEST (Paper Trading)")
    print("=" * 60)

    cm = ConnectionManager()

    print("\n[1] Connecting to IBKR...")
    if not cm.connect():
        print("FAILED: Could not connect")
        return False

    # Initialize managers
    om = OrderManager(cm)
    pm = PositionManager(cm, om)
    fetcher = DataFetcher(cm)

    # Step 2: Check current portfolio
    print("\n[2] Current Portfolio Status:")
    portfolio = pm.get_portfolio_value()
    print(f"    Net Liquidation: ${portfolio.get('net_liquidation', 0):,.2f}")
    print(f"    Buying Power:    ${portfolio.get('buying_power', 0):,.2f}")
    print(f"    Total Cash:      ${portfolio.get('total_cash', 0):,.2f}")

    # Step 3: Check current positions
    print("\n[3] Current Positions:")
    positions = pm.get_positions()
    if positions:
        for pos in positions:
            print(f"    {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
            print(f"      Market Value: ${pos.market_value:,.2f}")
            print(f"      Unrealized P&L: ${pos.unrealized_pnl:,.2f}")
    else:
        print("    No open positions")

    # Step 4: Check open orders
    print("\n[4] Open Orders:")
    open_orders = om.get_open_orders()
    if open_orders:
        for trade in open_orders:
            print(f"    Order {trade.order.orderId}: {trade.order.action} "
                  f"{trade.order.totalQuantity} {trade.contract.symbol} "
                  f"({trade.orderStatus.status})")
    else:
        print("    No open orders")

    # Step 5: Get current price for test symbol
    test_symbol = "AAPL"
    print(f"\n[5] Getting current price for {test_symbol}...")
    price = fetcher.get_latest_price(test_symbol)
    if price:
        print(f"    {test_symbol} last close: ${price:.2f}")
    else:
        print(f"    Could not get price for {test_symbol}")
        price = 250.0  # Fallback for testing

    # Step 6: Calculate position size
    print(f"\n[6] Position Sizing (10% of portfolio):")
    shares = pm.calculate_position_size(test_symbol, price, risk_pct=0.10)
    print(f"    Recommended size: {shares} shares")
    print(f"    Position value: ${shares * price:,.2f}")

    # Step 7: Test order placement (small test order)
    print("\n[7] Placing Test Order...")
    print("    Placing LIMIT BUY order for 1 share (below market for safety)")

    # Place limit order below current price so it won't fill immediately
    test_limit_price = round(price * 0.95, 2)  # 5% below market
    result = om.place_limit_order(
        symbol=test_symbol,
        action=OrderAction.BUY,
        quantity=1,
        limit_price=test_limit_price,
        reason="Test order - will cancel",
    )

    if result.success:
        print(f"    SUCCESS: Order {result.order_id} placed")
        print(f"    Limit price: ${test_limit_price}")

        # Step 8: Check order status
        print("\n[8] Checking Order Status...")
        cm.ib.sleep(1)  # Wait for status update
        status = om.get_order_status(result.order_id)
        print(f"    Order {result.order_id} status: {status}")

        # Step 9: Cancel the test order
        print("\n[9] Cancelling Test Order...")
        cancelled = om.cancel_order(result.order_id)
        if cancelled:
            print(f"    Order {result.order_id} cancelled successfully")
        else:
            print(f"    Failed to cancel order {result.order_id}")

    else:
        print(f"    FAILED: {result.message}")

    # Step 10: Show final state
    print("\n[10] Final State:")
    open_orders = om.get_open_orders()
    print(f"    Open orders: {len(open_orders)}")

    # Cleanup
    print("\n[11] Disconnecting...")
    cm.disconnect()

    print("\n" + "=" * 60)
    print("ORDER MANAGEMENT TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    print("\n⚠️  WARNING: This test will place REAL orders in your PAPER account.")
    print("    Orders will be cancelled immediately after placement.\n")

    response = input("Continue? (y/n): ").strip().lower()
    if response == 'y':
        success = test_orders()
        sys.exit(0 if success else 1)
    else:
        print("Test cancelled.")
        sys.exit(0)
