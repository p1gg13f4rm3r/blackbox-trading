"""
Chart pattern detection module.
Identifies common technical analysis patterns in OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class Pattern(ABC):
    """Base class for chart patterns."""

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect pattern in OHLCV data.

        Returns:
            List of detected patterns with metadata
        """
        pass


class PatternDetector:
    """Main pattern detection engine."""

    def __init__(self, sensitivity: float = 0.02):
        """
        Initialize detector.

        Args:
            sensitivity: Tolerance for pattern matching (e.g., 0.02 = 2%)
        """
        self.sensitivity = sensitivity
        self.patterns = []

    def register_pattern(self, pattern: Pattern):
        """Register a pattern detector."""
        self.patterns.append(pattern)

    def detect_all(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Detect all registered patterns.

        Returns:
            Dictionary mapping pattern names to detected instances
        """
        results = {}

        for pattern in self.patterns:
            detections = pattern.detect(df)
            if detections:
                pattern_name = pattern.__class__.__name__
                results[pattern_name] = detections
                print(f"✓ Found {len(detections)} {pattern_name} pattern(s)")

        return results


class HeadAndShoulders(Pattern):
    """Head and Shoulders pattern detection (bearish reversal)."""

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Head and Shoulders pattern.

        Structure:
        - Left shoulder (peak)
        - Head (higher peak)
        - Right shoulder (lower peak, similar to left)
        - Neckline support
        """
        detections = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Find local peaks
        peaks = self._find_peaks(highs, min_distance=5, prominence=0.02)

        # Need at least 3 peaks for H&S
        if len(peaks) < 3:
            return detections

        # Check each triplet of peaks
        for i in range(len(peaks) - 2):
            l_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            r_shoulder_idx = peaks[i + 2]

            l_shoulder = highs[l_shoulder_idx]
            head = highs[head_idx]
            r_shoulder = highs[r_shoulder_idx]

            # Head should be highest
            if not (head > l_shoulder and head > r_shoulder):
                continue

            # Shoulders should be roughly equal (within 5%)
            if abs(l_shoulder - r_shoulder) / max(l_shoulder, r_shoulder) > 0.05:
                continue

            # Find neckline (lowest point between peaks)
            neckline_range = df.iloc[l_shoulder_idx:r_shoulder_idx+1]
            neckline = neckline_range['low'].min()

            # Neckline at current price suggests breakdown
            current_price = closes[-1]
            if current_price < neckline * 0.98:  # Price broke below neckline
                detections.append({
                    'type': 'bearish_reversal',
                    'direction': 'sell',
                    'entry_idx': r_shoulder_idx,
                    'entry_price': current_price,
                    'stop_loss': head,
                    'target': neckline - (head - neckline),
                    'confidence': 0.7
                })

        return detections

    def _find_peaks(
        self,
        data: np.ndarray,
        min_distance: int = 5,
        prominence: float = 0.02
    ) -> np.ndarray:
        """Find local peaks using scipy."""
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            data,
            distance=min_distance,
            prominence=prominence * np.mean(data)
        )
        return peaks


class DoubleTop(Pattern):
    """Double Top pattern detection (bearish reversal)."""

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Double Top pattern.

        Structure:
        - Two roughly equal peaks
        - Neckline support between them
        """
        detections = []
        highs = df['high'].values

        peaks = self._find_peaks(highs, min_distance=10, prominence=0.03)

        if len(peaks) < 2:
            return detections

        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]

            peak1 = highs[peak1_idx]
            peak2 = highs[peak2_idx]

            # Peaks should be roughly equal (within 3%)
            if abs(peak1 - peak2) / max(peak1, peak2) > 0.03:
                continue

            # Find neckline (lowest point between peaks)
            neckline_range = df.iloc[peak1_idx:peak2_idx+1]
            neckline = neckline_range['low'].min()

            current_price = df['close'].values[-1]

            if current_price < neckline * 0.98:
                detections.append({
                    'type': 'bearish_reversal',
                    'direction': 'sell',
                    'entry_idx': peak2_idx,
                    'entry_price': current_price,
                    'stop_loss': max(peak1, peak2) * 1.01,
                    'target': neckline - (peak1 - neckline),
                    'confidence': 0.75
                })

        return detections

    def _find_peaks(
        self,
        data: np.ndarray,
        min_distance: int = 5,
        prominence: float = 0.02
    ) -> np.ndarray:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            data,
            distance=min_distance,
            prominence=prominence * np.mean(data)
        )
        return peaks


if __name__ == "__main__":
    # Test pattern detection
    from data import DataFetcher

    fetcher = DataFetcher()
    data = fetcher.fetch(["AAPL"], interval="1d", period="1y")

    detector = PatternDetector(sensitivity=0.02)
    detector.register_pattern(HeadAndShoulders())
    detector.register_pattern(DoubleTop())

    for symbol, df in data.items():
        print(f"\nAnalyzing {symbol}...")
        patterns = detector.detect_all(df)
        print(f"Found patterns: {list(patterns.keys())}")
