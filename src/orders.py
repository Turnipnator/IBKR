"""
Order management for IBKR trading.
Handles order placement, tracking, and position management.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ib_insync import (
    IB, Stock, Order, Trade, MarketOrder, LimitOrder,
    StopOrder, StopLimitOrder, BracketOrder
)

from .connection import ConnectionManager, get_connection
from .database import Database
from .config import trading_config

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[int] = None
    trade: Optional[Trade] = None
    message: str = ""
    fill_price: Optional[float] = None
    filled_quantity: int = 0


@dataclass
class Position:
    """Represents a current position."""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


class OrderManager:
    """
    Manages order placement and tracking for IBKR.

    Usage:
        om = OrderManager()
        result = om.place_market_order("AAPL", OrderAction.BUY, 10)
        if result.success:
            print(f"Order {result.order_id} placed")
    """

    def __init__(
        self,
        connection: Optional[ConnectionManager] = None,
        database: Optional[Database] = None,
    ):
        self.connection = connection or get_connection()
        self.db = database or Database()
        self._pending_orders: dict[int, Trade] = {}
        self._order_callbacks: list[Callable] = []

    @property
    def ib(self) -> IB:
        return self.connection.ib

    def _create_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Stock:
        """Create and qualify a stock contract."""
        contract = Stock(symbol, exchange, currency)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract for {symbol}")
        return qualified[0]

    def place_market_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        reason: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Stock ticker
            action: BUY or SELL
            quantity: Number of shares
            reason: Optional reason for the trade

        Returns:
            OrderResult with status and details
        """
        if not self.connection.ensure_connected():
            return OrderResult(success=False, message="Not connected to IBKR")

        try:
            contract = self._create_contract(symbol)
            order = MarketOrder(action.value, quantity)

            trade = self.ib.placeOrder(contract, order)
            self._pending_orders[trade.order.orderId] = trade

            logger.info(
                f"Placed market order: {action.value} {quantity} {symbol} "
                f"(orderId={trade.order.orderId})"
            )

            # Log to database
            self.db.save_trade(
                symbol=symbol,
                action=action.value,
                quantity=quantity,
                price=0.0,  # Market order, price unknown until filled
                order_id=trade.order.orderId,
                status="SUBMITTED",
                reason=reason,
            )

            return OrderResult(
                success=True,
                order_id=trade.order.orderId,
                trade=trade,
                message=f"Market order submitted: {action.value} {quantity} {symbol}",
            )

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return OrderResult(success=False, message=str(e))

    def place_limit_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        limit_price: float,
        reason: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            symbol: Stock ticker
            action: BUY or SELL
            quantity: Number of shares
            limit_price: Limit price
            reason: Optional reason for the trade

        Returns:
            OrderResult with status and details
        """
        if not self.connection.ensure_connected():
            return OrderResult(success=False, message="Not connected to IBKR")

        try:
            contract = self._create_contract(symbol)
            order = LimitOrder(action.value, quantity, limit_price)

            trade = self.ib.placeOrder(contract, order)
            self._pending_orders[trade.order.orderId] = trade

            logger.info(
                f"Placed limit order: {action.value} {quantity} {symbol} "
                f"@ ${limit_price} (orderId={trade.order.orderId})"
            )

            self.db.save_trade(
                symbol=symbol,
                action=action.value,
                quantity=quantity,
                price=limit_price,
                order_id=trade.order.orderId,
                status="SUBMITTED",
                reason=reason,
            )

            return OrderResult(
                success=True,
                order_id=trade.order.orderId,
                trade=trade,
                message=f"Limit order submitted: {action.value} {quantity} {symbol} @ ${limit_price}",
            )

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return OrderResult(success=False, message=str(e))

    def place_stop_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        stop_price: float,
        reason: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a stop order (for stop-loss).

        Args:
            symbol: Stock ticker
            action: BUY or SELL
            quantity: Number of shares
            stop_price: Stop trigger price
            reason: Optional reason

        Returns:
            OrderResult with status and details
        """
        if not self.connection.ensure_connected():
            return OrderResult(success=False, message="Not connected to IBKR")

        try:
            contract = self._create_contract(symbol)
            order = StopOrder(action.value, quantity, stop_price)

            trade = self.ib.placeOrder(contract, order)
            self._pending_orders[trade.order.orderId] = trade

            logger.info(
                f"Placed stop order: {action.value} {quantity} {symbol} "
                f"@ ${stop_price} (orderId={trade.order.orderId})"
            )

            self.db.save_trade(
                symbol=symbol,
                action=action.value,
                quantity=quantity,
                price=stop_price,
                order_id=trade.order.orderId,
                status="SUBMITTED",
                reason=reason or "Stop order",
            )

            return OrderResult(
                success=True,
                order_id=trade.order.orderId,
                trade=trade,
                message=f"Stop order submitted: {action.value} {quantity} {symbol} @ ${stop_price}",
            )

        except Exception as e:
            logger.error(f"Failed to place stop order: {e}")
            return OrderResult(success=False, message=str(e))

    def place_bracket_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        reason: Optional[str] = None,
    ) -> list[OrderResult]:
        """
        Place a bracket order (entry + take-profit + stop-loss).

        Args:
            symbol: Stock ticker
            action: BUY or SELL for entry
            quantity: Number of shares
            limit_price: Entry limit price
            take_profit_price: Take profit limit price
            stop_loss_price: Stop loss price

        Returns:
            List of OrderResults for each leg
        """
        if not self.connection.ensure_connected():
            return [OrderResult(success=False, message="Not connected to IBKR")]

        try:
            contract = self._create_contract(symbol)

            # Create bracket order
            bracket = self.ib.bracketOrder(
                action=action.value,
                quantity=quantity,
                limitPrice=limit_price,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price,
            )

            results = []
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                self._pending_orders[trade.order.orderId] = trade

                results.append(OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    trade=trade,
                    message=f"Bracket leg: {order.orderType} {order.action}",
                ))

            logger.info(
                f"Placed bracket order for {symbol}: entry=${limit_price}, "
                f"TP=${take_profit_price}, SL=${stop_loss_price}"
            )

            self.db.save_trade(
                symbol=symbol,
                action=action.value,
                quantity=quantity,
                price=limit_price,
                order_id=bracket[0].orderId,
                status="SUBMITTED",
                reason=reason or f"Bracket order (TP=${take_profit_price}, SL=${stop_loss_price})",
            )

            return results

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            return [OrderResult(success=False, message=str(e))]

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order by ID."""
        if not self.connection.ensure_connected():
            return False

        try:
            if order_id in self._pending_orders:
                trade = self._pending_orders[order_id]
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")
                return True
            else:
                # Try to find in open orders
                for trade in self.ib.openTrades():
                    if trade.order.orderId == order_id:
                        self.ib.cancelOrder(trade.order)
                        logger.info(f"Cancelled order {order_id}")
                        return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        if not self.connection.ensure_connected():
            return 0

        cancelled = 0
        for trade in self.ib.openTrades():
            try:
                self.ib.cancelOrder(trade.order)
                cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {trade.order.orderId}: {e}")

        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    def get_open_orders(self) -> list[Trade]:
        """Get all open orders."""
        if not self.connection.ensure_connected():
            return []
        return self.ib.openTrades()

    def get_order_status(self, order_id: int) -> Optional[str]:
        """Get the status of an order."""
        if not self.connection.ensure_connected():
            return None

        for trade in self.ib.trades():
            if trade.order.orderId == order_id:
                return trade.orderStatus.status

        return None

    def wait_for_fill(
        self,
        order_id: int,
        timeout: float = 30.0,
    ) -> OrderResult:
        """
        Wait for an order to fill.

        Args:
            order_id: The order ID to wait for
            timeout: Maximum seconds to wait

        Returns:
            OrderResult with fill details
        """
        if not self.connection.ensure_connected():
            return OrderResult(success=False, message="Not connected")

        trade = self._pending_orders.get(order_id)
        if not trade:
            for t in self.ib.trades():
                if t.order.orderId == order_id:
                    trade = t
                    break

        if not trade:
            return OrderResult(success=False, message=f"Order {order_id} not found")

        # Wait for fill
        start = datetime.now()
        while (datetime.now() - start).total_seconds() < timeout:
            self.ib.sleep(0.5)

            if trade.orderStatus.status == 'Filled':
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    trade=trade,
                    fill_price=trade.orderStatus.avgFillPrice,
                    filled_quantity=int(trade.orderStatus.filled),
                    message=f"Filled @ ${trade.orderStatus.avgFillPrice}",
                )
            elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled']:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message="Order was cancelled",
                )

        return OrderResult(
            success=False,
            order_id=order_id,
            message=f"Timeout waiting for fill (status: {trade.orderStatus.status})",
        )


class PositionManager:
    """
    Manages current positions and portfolio.

    Usage:
        pm = PositionManager()
        positions = pm.get_positions()
        pm.close_position("AAPL")
    """

    def __init__(
        self,
        connection: Optional[ConnectionManager] = None,
        order_manager: Optional[OrderManager] = None,
    ):
        self.connection = connection or get_connection()
        self.order_manager = order_manager or OrderManager(self.connection)

    @property
    def ib(self) -> IB:
        return self.connection.ib

    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        if not self.connection.ensure_connected():
            return []

        positions = []
        for pos in self.ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                quantity=int(pos.position),
                avg_cost=pos.avgCost,
                market_value=0.0,  # Updated below if portfolio available
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            ))

        # Enrich with portfolio data
        for portfolio_item in self.ib.portfolio():
            for pos in positions:
                if pos.symbol == portfolio_item.contract.symbol:
                    pos.market_value = portfolio_item.marketValue
                    pos.unrealized_pnl = portfolio_item.unrealizedPNL
                    pos.realized_pnl = portfolio_item.realizedPNL

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        pos = self.get_position(symbol)
        return pos is not None and pos.quantity != 0

    def get_position_quantity(self, symbol: str) -> int:
        """Get the quantity of shares held for a symbol."""
        pos = self.get_position(symbol)
        return pos.quantity if pos else 0

    def close_position(
        self,
        symbol: str,
        reason: Optional[str] = None,
    ) -> OrderResult:
        """
        Close an entire position in a symbol.

        Args:
            symbol: Stock ticker
            reason: Optional reason for closing

        Returns:
            OrderResult with details
        """
        pos = self.get_position(symbol)
        if not pos or pos.quantity == 0:
            return OrderResult(
                success=False,
                message=f"No position in {symbol}",
            )

        action = OrderAction.SELL if pos.quantity > 0 else OrderAction.BUY
        quantity = abs(pos.quantity)

        return self.order_manager.place_market_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            reason=reason or "Close position",
        )

    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions."""
        results = []
        for pos in self.get_positions():
            if pos.quantity != 0:
                result = self.close_position(pos.symbol, "Close all positions")
                results.append(result)
        return results

    def get_portfolio_value(self) -> dict:
        """Get portfolio summary values."""
        if not self.connection.ensure_connected():
            return {}

        summary = self.connection.get_account_summary()
        return {
            'net_liquidation': float(summary.get('NetLiquidation', {}).get('value', 0)),
            'total_cash': float(summary.get('TotalCashValue', {}).get('value', 0)),
            'buying_power': float(summary.get('BuyingPower', {}).get('value', 0)),
            'unrealized_pnl': float(summary.get('UnrealizedPnL', {}).get('value', 0)),
            'realized_pnl': float(summary.get('RealizedPnL', {}).get('value', 0)),
        }

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_pct: Optional[float] = None,
    ) -> int:
        """
        Calculate position size based on portfolio and risk parameters.

        Args:
            symbol: Stock ticker
            price: Current price per share
            risk_pct: Max portfolio percentage for position (default from config)

        Returns:
            Number of shares to buy
        """
        if not self.connection.ensure_connected():
            return 0

        risk_pct = risk_pct or trading_config.max_position_pct
        portfolio = self.get_portfolio_value()
        net_liq = portfolio.get('net_liquidation', 0)

        if net_liq <= 0 or price <= 0:
            return 0

        max_position_value = net_liq * risk_pct
        shares = int(max_position_value / price)

        return max(0, shares)
