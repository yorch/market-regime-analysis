#!/usr/bin/env python3
"""
Jim Simons Market Regime Analysis System - Main Application

This is the main entry point for the comprehensive market regime analysis system
implementing Jim Simons' Hidden Markov Model methodology for quantitative trading.
"""

import os
import sys

from market_regime_analysis import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)
from market_regime_analysis.data_provider import AlphaVantageProvider

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MarketRegimeApp:
    """
    Main application class providing interactive menu system.

    This class implements the user interface for the market regime analysis
    system following the exact specifications in PLAN.md.
    """

    def __init__(self) -> None:
        """Initialize the application."""
        self.current_analyzer: MarketRegimeAnalyzer | None = None
        self.current_portfolio: PortfolioHMMAnalyzer | None = None

    def print_header(self) -> None:
        """Print application header."""
        print("\n" + "=" * 80)
        print("JIM SIMONS MARKET REGIME ANALYSIS SYSTEM")
        print("Hidden Markov Model Implementation")
        print("Following Renaissance Technologies Methodology")
        print("=" * 80)
        print(
            "Data Provider Options: Yahoo Finance (yfinance), Alpha Vantage (recommended)"
        )
        print(
            "Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY env variable or enter at prompt."
        )
        print("=" * 80)

    def print_menu(self) -> None:
        """Print the main menu options."""
        print("\n📊 MAIN MENU:")
        print("1. Current HMM Regime Analysis (All Timeframes)")
        print("2. Detailed HMM Analysis (Single Timeframe)")
        print("3. Generate HMM Charts")
        print("4. Export HMM Analysis to CSV")
        print("5. Start Continuous HMM Monitoring")
        print("6. Multi-Symbol HMM Analysis")
        print("7. Position Sizing Calculator")
        print("8. Exit")
        print("-" * 50)

    def get_symbol_input(self, default: str = "SPY") -> str:
        """
        Get symbol input from user.

        Args:
            default: Default symbol if user presses enter

        Returns:
            Trading symbol
        """
        symbol = input(f"Enter symbol (default: {default}): ").strip().upper()
        return symbol if symbol else default

    def get_timeframe_input(self) -> str:
        """
        Get timeframe input from user.

        Returns:
            Selected timeframe
        """
        print("\nAvailable timeframes:")
        print("1. 1D (Daily)")
        print("2. 1H (Hourly)")
        print("3. 15m (15-minute)")

        while True:
            choice = input("Select timeframe (1-3): ").strip()
            if choice == "1":
                return "1D"
            elif choice == "2":
                return "1H"
            elif choice == "3":
                return "15m"
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

    def get_provider_input(self) -> tuple[str, str | None]:
        """
        Prompt user to select data provider and enter API key if needed.

        Returns:
            provider_flag: 'yfinance' or 'alphavantage'
            api_key: API key for Alpha Vantage or None
        """
        print("\nSelect data provider:")
        print("1. Yahoo Finance (yfinance)")
        print("2. Alpha Vantage (recommended)")
        while True:
            choice = input("Provider (1-2, default 2): ").strip()
            if choice in ["", "2"]:
                provider_flag = "alphavantage"
                api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
                if not api_key:
                    api_key = input("Enter Alpha Vantage API key: ").strip()
                if not api_key or not isinstance(api_key, str):
                    print("API key required for Alpha Vantage.")
                    continue
                return provider_flag, api_key
            elif choice == "1":
                return "yfinance", None
            else:
                print("Invalid choice. Please select 1 or 2.")

    def option_1_current_analysis(self) -> None:
        """Option 1: Current HMM Regime Analysis (All Timeframes)."""
        print("\n" + "=" * 60)
        print("CURRENT HMM REGIME ANALYSIS (ALL TIMEFRAMES)")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        symbol = self.get_symbol_input()
        try:
            print(f"\nInitializing analyzer for {symbol}...")
            if provider_flag == "alphavantage":
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
                analyzer.provider = AlphaVantageProvider(api_key)
            else:
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
            self.current_analyzer = analyzer

            # Analyze all timeframes
            for timeframe in ["1D", "1H", "15m"]:
                try:
                    print(f"\n{'-'*40}")
                    print(f"ANALYSIS FOR {timeframe}")
                    print(f"{'-'*40}")
                    analyzer.print_analysis_report(timeframe)
                except Exception as e:
                    print(f"Error analyzing {timeframe}: {e!s}")

        except Exception as e:
            print(f"Error: {e!s}")

    def option_2_detailed_analysis(self) -> None:
        """Option 2: Detailed HMM Analysis (Single Timeframe)."""
        print("\n" + "=" * 60)
        print("DETAILED HMM ANALYSIS (SINGLE TIMEFRAME)")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        symbol = self.get_symbol_input()
        timeframe = self.get_timeframe_input()

        try:
            print(f"\nInitializing detailed analysis for {symbol} ({timeframe})...")
            if provider_flag == "alphavantage":
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
                analyzer.provider = AlphaVantageProvider(api_key)
            else:
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
            self.current_analyzer = analyzer

            # Get detailed analysis
            analysis = analyzer.analyze_current_regime(timeframe)

            # Print comprehensive report
            analyzer.print_analysis_report(timeframe)

            # Additional detailed information
            print("\n📋 DETAILED METRICS:")
            print(f"   HMM State: {analysis.hmm_state}/5")
            print(f"   Regime: {analysis.current_regime.value}")
            print(f"   Confidence: {analysis.regime_confidence:.3f}")
            print(f"   Persistence: {analysis.regime_persistence:.3f}")
            print(f"   Transition Prob: {analysis.transition_probability:.3f}")

            # Risk metrics
            print("\n⚠️  RISK ASSESSMENT:")
            print(f"   Risk Level: {analysis.risk_level}")
            print(f"   Position Multiplier: {analysis.position_sizing_multiplier:.3f}")
            print(f"   Strategy: {analysis.recommended_strategy.value}")

        except Exception as e:
            print(f"Error: {e!s}")

    def option_3_generate_charts(self) -> None:
        """Option 3: Generate HMM Charts."""
        print("\n" + "=" * 60)
        print("GENERATE HMM CHARTS")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        if self.current_analyzer is None:
            symbol = self.get_symbol_input()
            try:
                print(f"Initializing analyzer for {symbol}...")
                if provider_flag == "alphavantage":
                    self.current_analyzer = MarketRegimeAnalyzer(
                        symbol, provider_flag=provider_flag
                    )
                    self.current_analyzer.provider = AlphaVantageProvider(api_key)
                else:
                    self.current_analyzer = MarketRegimeAnalyzer(
                        symbol, provider_flag=provider_flag
                    )
            except Exception as e:
                print(f"Error initializing analyzer: {e!s}")
                return

        timeframe = self.get_timeframe_input()

        try:
            days = input("Enter number of days to plot (default: 60): ").strip()
            days = int(days) if days.isdigit() else 60

            print(f"Generating charts for {timeframe} ({days} days)...")
            self.current_analyzer.plot_regime_analysis(timeframe, days)

        except Exception as e:
            print(f"Error generating charts: {e!s}")

    def option_4_export_csv(self) -> None:
        """Option 4: Export HMM Analysis to CSV."""
        print("\n" + "=" * 60)
        print("EXPORT HMM ANALYSIS TO CSV")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        if self.current_analyzer is None:
            symbol = self.get_symbol_input()
            try:
                print(f"Initializing analyzer for {symbol}...")
                if provider_flag == "alphavantage":
                    self.current_analyzer = MarketRegimeAnalyzer(
                        symbol, provider_flag=provider_flag
                    )
                    self.current_analyzer.provider = AlphaVantageProvider(api_key)
                else:
                    self.current_analyzer = MarketRegimeAnalyzer(
                        symbol, provider_flag=provider_flag
                    )
            except Exception as e:
                print(f"Error initializing analyzer: {e!s}")
                return

        try:
            filename = input(
                "Enter filename (press Enter for auto-generated): "
            ).strip()
            filename = filename if filename else None

            print("Exporting analysis data...")
            self.current_analyzer.export_analysis_to_csv(filename)

        except Exception as e:
            print(f"Error exporting data: {e!s}")

    def option_5_continuous_monitoring(self) -> None:
        """Option 5: Start Continuous HMM Monitoring."""
        print("\n" + "=" * 60)
        print("CONTINUOUS HMM MONITORING")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        symbol = self.get_symbol_input()
        try:
            interval_str = input(
                "Enter refresh interval in seconds (default: 300): "
            ).strip()
            interval = int(interval_str) if interval_str.isdigit() else 300
            print(f"Starting continuous monitoring for {symbol}...")
            if provider_flag == "alphavantage":
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
                analyzer.provider = AlphaVantageProvider(api_key)
            else:
                analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_flag)
            analyzer.run_continuous_monitoring(interval)
        except Exception as e:
            print(f"Error in continuous monitoring: {e!s}")

    def option_6_multi_symbol_analysis(self) -> None:
        """Option 6: Multi-Symbol HMM Analysis."""
        print("\n" + "=" * 60)
        print("MULTI-SYMBOL HMM ANALYSIS")
        print("=" * 60)

        provider_flag, api_key = self.get_provider_input()
        print("Enter symbols separated by commas (e.g., SPY,QQQ,IWM):")
        symbols_input = input("Symbols: ").strip().upper()
        if not symbols_input:
            symbols = ["SPY", "QQQ", "IWM"]
            print("Using default symbols: SPY, QQQ, IWM")
        else:
            symbols = [s.strip() for s in symbols_input.split(",")]
        try:
            print(f"Initializing portfolio analysis for {len(symbols)} symbols...")
            if provider_flag == "alphavantage":
                portfolio = PortfolioHMMAnalyzer(
                    symbols, provider_flag=provider_flag, api_key=api_key
                )
            else:
                portfolio = PortfolioHMMAnalyzer(symbols, provider_flag=provider_flag)
            self.current_portfolio = portfolio

            # Print comprehensive portfolio analysis
            portfolio.print_portfolio_summary("1D")

        except Exception as e:
            print(f"Error in portfolio analysis: {e!s}")

    def option_7_position_sizing(self) -> None:
        """Option 7: Position Sizing Calculator."""
        print("\n" + "=" * 60)
        print("POSITION SIZING CALCULATOR")
        print("=" * 60)

        try:
            # Get inputs
            base_size = float(input("Enter base position size (0.01-1.0): ") or "0.02")

            print("\nSelect regime:")
            regimes = list(MarketRegime)
            for i, regime in enumerate(regimes, 1):
                print(f"{i}. {regime.value}")

            regime_choice = int(input("Select regime (1-7): ") or "1") - 1
            regime = regimes[regime_choice]

            confidence = float(input("Enter regime confidence (0.0-1.0): ") or "0.8")
            persistence = float(input("Enter regime persistence (0.0-1.0): ") or "0.7")
            correlation = float(
                input("Enter portfolio correlation (-1.0-1.0): ") or "0.0"
            )

            # Calculate position size
            size = SimonsRiskCalculator.calculate_regime_adjusted_size(
                base_size, regime, confidence, persistence
            )

            correlation_adjusted = (
                SimonsRiskCalculator.calculate_correlation_adjusted_size(
                    size, correlation
                )
            )

            print("\n📊 POSITION SIZING RESULTS:")
            print(f"   Base Size: {base_size:.1%}")
            print(f"   Regime: {regime.value}")
            print(f"   Regime Adjusted: {size:.1%}")
            print(f"   Correlation Adjusted: {correlation_adjusted:.1%}")
            print(f"   Final Recommendation: {correlation_adjusted:.1%}")

        except Exception as e:
            print(f"Error in position sizing calculation: {e!s}")

    def run(self) -> None:
        """Main application loop."""
        while True:
            try:
                self.print_header()
                self.print_menu()

                choice = input("Select option (1-8): ").strip()

                if choice == "1":
                    self.option_1_current_analysis()
                elif choice == "2":
                    self.option_2_detailed_analysis()
                elif choice == "3":
                    self.option_3_generate_charts()
                elif choice == "4":
                    self.option_4_export_csv()
                elif choice == "5":
                    self.option_5_continuous_monitoring()
                elif choice == "6":
                    self.option_6_multi_symbol_analysis()
                elif choice == "7":
                    self.option_7_position_sizing()
                elif choice == "8":
                    print("\nThank you for using the Market Regime Analysis System!")
                    break
                else:
                    print("\nInvalid choice. Please select 1-8.")

                # Pause before showing menu again
                if choice in ["1", "2", "3", "4", "6", "7"]:
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\nExiting application...")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e!s}")
                input("Press Enter to continue...")


def main() -> None:
    """Main entry point."""
    try:
        app = MarketRegimeApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Fatal error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
