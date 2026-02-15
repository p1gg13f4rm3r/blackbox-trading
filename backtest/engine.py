"""
Backtesting module.
Test trading strategies on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class Trade:
    """Represents a single trade."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Calculate P&L
        if direction == 'buy':
            self.pnl = (exit_price - entry_price) * quantity
        else:
            self.pnl = (entry_price - exit_price) * quantity

        self.pnl_pct = (self.pnl / (entry_price * quantity)) * 100
        self.duration = (exit_time - entry_time).days

    def to_dict(self) -> Dict:
        """Convert trade to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'duration_days': self.duration,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }

    def __repr__(self) -> str:
        return (
            f"Trade({self.symbol} {self.direction} | "
            f"Entry: {self.entry_price:.2f} Exit: {self.exit_price:.2f} | "
            f"P&L: {self.pnl:.2f} ({self.pnl_pct:.2f}%) | "
            f"{self.duration} days)"
        )


class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0001
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as decimal, e.g., 0.001 = 0.1%)
            slippage: Slippage per trade (as decimal, e.g., 0.0001 = 0.01%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.capital = initial_capital
        self.trades: List[Trade] = []

    def run(
        self,
        signals: List,
        data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run backtest on signals with historical data.

        Args:
            signals: List of Signal objects
            data: Dictionary mapping symbols to OHLCV DataFrames

        Returns:
            Dictionary with backtest results
        """
        self.trades = []
        self.capital = self.initial_capital

        for signal in signals:
            symbol = signal.symbol
            df = data.get(symbol)

            if df is None or signal.entry_idx >= len(df):
                continue

            # Get trade details
            entry_price = signal.entry_price
            quantity = self._calculate_position_size(entry_price)

            # Simulate trade
            trade = self._simulate_trade(signal, df, quantity)
            if trade:
                self.trades.append(trade)
                self.capital += trade.pnl

        return self._calculate_results()

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on available capital."""
        position_value = self.capital * 0.95  # Use 95% of capital per trade
        return position_value / price

    def _simulate_trade(
        self,
        signal,
        df: pd.DataFrame,
        quantity: float
    ) -> Optional[Trade]:
        """Simulate a single trade from signal to exit."""
        entry_idx = signal.entry_idx
        entry_price = signal.entry_price

        # Apply slippage to entry
        if signal.direction == 'buy':
            entry_price *= (1 + self.slippage)
        else:
            entry_price *= (1 - self.slippage)

        entry_time = df.index[entry_idx]

        # Find exit
        exit_price, exit_idx = self._find_exit(signal, df, entry_idx)

        if exit_idx is None:
            # No exit found, use last available price
            exit_idx = len(df) - 1
            exit_price = df['close'].iloc[exit_idx]

        exit_time = df.index[exit_idx]

        # Calculate commission
        commission_cost = (entry_price + exit_price) * quantity * self.commission

        # Create trade
        trade = Trade(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.target
        )

        # Adjust for commission
        trade.pnl -= commission_cost

        return trade

    def _find_exit(
        self,
        signal,
        df: pd.DataFrame,
        start_idx: int
    ) -> tuple:
        """Find exit point based on stop loss or take profit."""
        highs = df['high'].values[start_idx:]
        lows = df['low'].values[start_idx:]
        closes = df['close'].values[start_idx:]

        for i in range(len(highs)):
            if signal.direction == 'buy':
                if lows[i] <= signal.stop_loss:
                    return signal.stop_loss, start_idx + i
                elif highs[i] >= signal.target:
                    return signal.target, start_idx + i
            else:
                if highs[i] >= signal.stop_loss:
                    return signal.stop_loss, start_idx + i
                elif lows[i] <= signal.target:
                    return signal.target, start_idx + i

        return None, None

    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'final_capital': self.capital,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        # Basic metrics
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / total_trades

        # Returns
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        # Average win/loss
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate max drawdown
        equity_curve = self._calculate_equity_curve()
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'total_return': total_return,
            'final_capital': self.capital,
            'initial_capital': self.initial_capital,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': np.mean([t.pnl for t in self.trades])
        }

    def _calculate_equity_curve(self) -> List[float]:
        """Calculate equity curve over time."""
        equity = [self.initial_capital]
        running_capital = self.initial_capital

        for trade in sorted(self.trades, key=lambda t: t.entry_time):
            running_capital += trade.pnl
            equity.append(running_capital)

        return equity

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)

        return max_dd

    def print_summary(self):
        """Print backtest results summary."""
        results = self._calculate_results()

        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Win Rate:            {results['win_rate']:.2f}%")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Total Return:        {results['total_return']:.2f}%")
        print(f"Initial Capital:     ${results['initial_capital']:.2f}")
        print(f"Final Capital:       ${results['final_capital']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']:.2f}%")
        print(f"Avg Win:             ${results['avg_win']:.2f}")
        print(f"Avg Loss:            ${results['avg_loss']:.2f}")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Avg Trade:           ${results['avg_trade']:.2f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test backtester
    backtester = Backtester(initial_capital=10000, commission=0.001)
    print("Backtester initialized successfully!")
