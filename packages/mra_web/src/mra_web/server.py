#!/usr/bin/env python3
"""
Startup script for the Market Regime Analysis API server.

This script provides an easy way to start the API server with proper configuration.
"""

import argparse
import os
import sys

import uvicorn

from mra_web.config import config


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Market Regime Analysis API Server")

    parser.add_argument("--host", default=config.host, help="Host to bind to (default: 0.0.0.0)")

    parser.add_argument(
        "--port", type=int, default=config.port, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=config.reload,
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--workers", type=int, default=config.workers, help="Number of worker processes"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=config.log_level,
        help="Logging level",
    )

    parser.add_argument(
        "--dev", action="store_true", help="Run in development mode (enables reload, debug, etc.)"
    )

    args = parser.parse_args()

    # Development mode overrides
    if args.dev:
        args.reload = True
        args.workers = 1
        args.log_level = "DEBUG"
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"
        print("🚀 Starting in DEVELOPMENT mode")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"📊 Health Check: http://{args.host}:{args.port}/health")
        print(f"📈 Metrics: http://{args.host}:{args.port}/metrics")

    # Show configuration
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"⚡ Workers: {args.workers}")
    print(f"📝 Log Level: {args.log_level}")
    print(f"🔄 Reload: {args.reload}")

    # Environment check
    if args.host == "0.0.0.0" and not args.dev:
        print("⚠️  WARNING: Binding to 0.0.0.0 in production. Ensure proper firewall configuration.")

    # API key reminders
    print("\n🔑 API Key Configuration:")
    print("   Alpha Vantage: Set ALPHA_VANTAGE_API_KEY environment variable")
    print("   Polygon.io: Set POLYGON_API_KEY environment variable")
    print("   Yahoo Finance: No API key required (free tier)")

    try:
        # Start server
        uvicorn.run(
            "mra_web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
            access_log=True,
        )

    except KeyboardInterrupt:
        print("\n👋 Shutting down API server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
