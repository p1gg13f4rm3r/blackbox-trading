"""
Black Box Trading - Main Entry Point
Orchestrates data fetching, pattern detection, signal generation, and backtesting.
"""

import yaml
import argparse
from pathlib import Path

from data import DataFetcher
from patterns import PatternDetector, HeadAndShoulders, DoubleTop
from signals import SignalGenerator, SignalTracker
from backtest import Backtester


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Black Box Trading Framework")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "backtest", "live"],
        default="scan",
        help="Operation mode: scan, backtest, or live"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")

    # Initialize components
    fetcher = DataFetcher(provider=config['data']['provider'])

    detector = PatternDetector(sensitivity=config['patterns']['sensitivity'])

    # Register enabled patterns
    pattern_map = {
        'head_and_shoulders': HeadAndShoulders,
        'double_top': DoubleTop,
    }

    for pattern_name in config['patterns']['enable']:
        if pattern_name in pattern_map:
            detector.register_pattern(pattern_map[pattern_name]())
            print(f"✓ Registered pattern: {pattern_name}")

    signal_gen = SignalGenerator(
        min_confidence=config['signals']['min_confidence'],
        stop_loss_pct=config['signals']['stop_loss_pct'],
        take_profit_pct=config['signals']['take_profit_pct']
    )

    backtester = Backtester(
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission'],
        slippage=config['backtest']['slippage']
    )

    # Fetch data
    print("\n" + "="*50)
    print("FETCHING MARKET DATA")
    print("="*50)
    data = fetcher.fetch(
        symbols=config['data']['symbols'],
        interval=config['data']['interval'],
        period=config['data']['period']
    )

    if not data:
        print("✗ No data fetched. Exiting.")
        return

    # Run based on mode
    if args.mode == "scan":
        print("\n" + "="*50)
        print("SCANNING FOR PATTERNS")
        print("="*50)

        all_signals = []

        for symbol, df in data.items():
            print(f"\nAnalyzing {symbol}...")

            # Detect patterns
            patterns = detector.detect_all(df)

            if patterns:
                # Generate signals
                signals = signal_gen.generate(symbol, df, patterns)
                filtered = signal_gen.filter_signals(signals)

                all_signals.extend(filtered)

                # Print signals
                for signal in filtered:
                    print(f"  {signal}")

        # Summary
        print(f"\n{'='*50}")
        print(f"SUMMARY: Found {len(all_signals)} signals")
        print(f"{'='*50}")

        # Save signals if configured
        if config['output']['save_signals']:
            save_signals(all_signals)

    elif args.mode == "backtest":
        print("\n" + "="*50)
        print("BACKTESTING STRATEGY")
        print("="*50)

        all_signals = []

        for symbol, df in data.items():
            patterns = detector.detect_all(df)
            signals = signal_gen.generate(symbol, df, patterns)
            all_signals.extend(signals)

        # Run backtest
        results = backtester.run(all_signals, data)
        backtester.print_summary()

        # Save results if configured
        if config['output']['save_backtest_results']:
            save_backtest_results(results)

    print("\n✓ Done!")


def save_signals(signals: list):
    """Save signals to CSV file."""
    import pandas as pd
    from datetime import datetime

    signal_dicts = [s.to_dict() for s in signals]
    df = pd.DataFrame(signal_dicts)

    filename = f"signals/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    Path("signals").mkdir(exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(signals)} signals to {filename}")


def save_backtest_results(results: dict):
    """Save backtest results to file."""
    import json
    from datetime import datetime

    filename = f"backtest/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("backtest").mkdir(exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved backtest results to {filename}")


if __name__ == "__main__":
    main()
