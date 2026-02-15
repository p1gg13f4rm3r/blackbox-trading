"""
Signal generation module.
Converts pattern detections into actionable trading signals.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class Signal:
    """Represents a trading signal."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        pattern_type: str,
        timestamp: pd.Timestamp
    ):
        self.symbol = symbol
        self.direction = direction  # 'buy' or 'sell'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.confidence = confidence
        self.pattern_type = pattern_type
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'confidence': self.confidence,
            'pattern_type': self.pattern_type,
            'timestamp': self.timestamp.isoformat(),
            'risk_reward': abs(self.target - self.entry_price) / abs(self.entry_price - self.stop_loss)
        }

    def __repr__(self) -> str:
        return (
            f"Signal({self.symbol} {self.direction.upper()} @ {self.entry_price:.2f} | "
            f"SL: {self.stop_loss:.2f} | TP: {self.target:.2f} | "
            f"Conf: {self.confidence:.2f} | {self.pattern_type})"
        )


class SignalGenerator:
    """Generate trading signals from pattern detections."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Initialize signal generator.

        Args:
            min_confidence: Minimum confidence threshold for signals
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
        """
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        patterns: Dict[str, List[Dict]]
    ) -> List[Signal]:
        """
        Generate signals from detected patterns.

        Args:
            symbol: Trading symbol
            df: OHLCV data
            patterns: Detected patterns from PatternDetector

        Returns:
            List of Signal objects
        """
        signals = []

        for pattern_name, pattern_list in patterns.items():
            for pattern_data in pattern_list:
                # Filter by confidence
                if pattern_data.get('confidence', 0) < self.min_confidence:
                    continue

                # Create signal
                signal = Signal(
                    symbol=symbol,
                    direction=pattern_data['direction'],
                    entry_price=pattern_data['entry_price'],
                    stop_loss=pattern_data.get('stop_loss'),
                    target=pattern_data.get('target'),
                    confidence=pattern_data.get('confidence'),
                    pattern_type=pattern_name,
                    timestamp=df.index[pattern_data['entry_idx']]
                )

                signals.append(signal)

        return signals

    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply additional filters to signals.

        Can be extended with:
        - Volume confirmation
        - Trend filters
        - Market regime detection
        - Volatility filters
        """
        filtered = []

        for signal in signals:
            # Example filter: Minimum risk-reward ratio of 1:2
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.target - signal.entry_price)

            if risk > 0 and (reward / risk) >= 2.0:
                filtered.append(signal)

        return filtered


class SignalTracker:
    """Track and manage active signals."""

    def __init__(self):
        self.active_signals = {}
        self.closed_signals = []

    def add_signal(self, signal: Signal):
        """Add a new signal to track."""
        signal_id = f"{signal.symbol}_{signal.timestamp.timestamp()}"
        self.active_signals[signal_id] = signal
        print(f"✓ Added signal: {signal}")

    def update_signals(self, current_price: float):
        """Update signal statuses based on current price."""
        to_close = []

        for signal_id, signal in self.active_signals.items():
            # Check stop loss
            if signal.direction == 'buy' and current_price <= signal.stop_loss:
                signal.status = 'stopped_out'
                to_close.append(signal_id)
            elif signal.direction == 'sell' and current_price >= signal.stop_loss:
                signal.status = 'stopped_out'
                to_close.append(signal_id)

            # Check take profit
            elif signal.direction == 'buy' and current_price >= signal.target:
                signal.status = 'target_hit'
                to_close.append(signal_id)
            elif signal.direction == 'sell' and current_price <= signal.target:
                signal.status = 'target_hit'
                to_close.append(signal_id)

        # Move closed signals
        for signal_id in to_close:
            signal = self.active_signals.pop(signal_id)
            self.closed_signals.append(signal)
            print(f"✓ Closed signal: {signal} - {signal.status}")


if __name__ == "__main__":
    # Test signal generation
    generator = SignalGenerator(min_confidence=0.7)

    # Example pattern detection
    patterns = {
        'HeadAndShoulders': [
            {
                'direction': 'sell',
                'entry_price': 150.0,
                'stop_loss': 155.0,
                'target': 140.0,
                'confidence': 0.75,
                'entry_idx': 50
            }
        ]
    }

    # Mock data
    df = pd.DataFrame({
        'close': [150.0] * 100
    }, index=pd.date_range('2024-01-01', periods=100))

    signals = generator.generate('AAPL', df, patterns)
    for signal in signals:
        print(signal)
