"""
Market data fetcher module.
Supports multiple data providers: yfinance, Alpha Vantage, Polygon.io
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch and store market data from various sources."""

    def __init__(self, provider: str = "yfinance"):
        self.provider = provider
        self._fetchers = {
            "yfinance": self._fetch_yfinance,
            # Add more providers as needed
        }

    def fetch(
        self,
        symbols: List[str],
        interval: str = "1d",
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for given symbols.

        Args:
            symbols: List of ticker symbols
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk)
            period: How much historical data to fetch

        Returns:
            Dictionary mapping symbols to OHLCV DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._fetchers[self.provider](symbol, interval, period)
                if data is not None and not data.empty:
                    results[symbol] = data
                    print(f"✓ Fetched {len(data)} bars for {symbol}")
                else:
                    print(f"✗ No data for {symbol}")
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {e}")

        return results

    def _fetch_yfinance(
        self,
        symbol: str,
        interval: str,
        period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance."""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return None

        # Standardize column names
        data.columns = data.columns.str.lower()
        return data

    def save_to_csv(self, data: Dict[str, pd.DataFrame], path: str = "data/"):
        """Save fetched data to CSV files."""
        import os
        os.makedirs(path, exist_ok=True)

        for symbol, df in data.items():
            filename = f"{path}{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename)
            print(f"✓ Saved {symbol} to {filename}")


if __name__ == "__main__":
    # Test the fetcher
    fetcher = DataFetcher()
    data = fetcher.fetch(["AAPL", "TSLA"], interval="1d", period="3mo")

    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(df.head())
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
