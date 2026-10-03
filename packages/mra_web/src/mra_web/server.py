#!/usr/bin/env python3
"""
Startup script for the Market Regime Analysis API server.

This script provides an easy way to start the API server with proper configuration.
The configuration is validated before uvicorn starts, so an unsafe setup (e.g. no
``JWT_SECRET`` in production) exits with a clear message instead of a traceback.
"""

import argparse
import os
import sys

import uvicorn


def main() -> None:
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Market Regime Analysis API Server")

    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host to bind to (default: API_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to (default: API_PORT or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "false").lower() == "true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.getenv("LOG_LEVEL", "INFO").upper(),
        help="Logging level",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (ENVIRONMENT=development, reload, debug logging)",
    )

    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        dev=args.dev,
    )


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "INFO",
    dev: bool = False,
) -> None:
    """Validate the configuration and run the API server under uvicorn.

    Shared by ``mra-api`` and ``mra start-api``. With ``dev=True`` it sets
    ``ENVIRONMENT=development`` and ``DEBUG=true`` (so no ``JWT_SECRET`` is needed),
    enables reload, and logs at DEBUG.
    """
    log_level = log_level.upper()
    # Development mode overrides (must be set before the config is loaded)
    if dev:
        reload = True
        workers = 1
        log_level = "DEBUG"
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = log_level

    from mra_web.config import APIConfig, ConfigError  # noqa: PLC0415

    try:
        cfg = APIConfig.from_env()
    except (ConfigError, ValueError) as e:
        print(f"❌ Refusing to start: {e}", file=sys.stderr)
        sys.exit(2)

    if cfg.is_development:
        print("🚀 Starting in DEVELOPMENT mode (unauthenticated requests allowed)")
        if cfg.docs_enabled:
            print(f"📖 API Documentation: http://{host}:{port}/docs")
        print(f"📊 Health Check: http://{host}:{port}/health")

    if cfg.jwt_secret_ephemeral and workers > 1:
        print("❌ An ephemeral JWT secret cannot be shared across workers; set JWT_SECRET.")
        sys.exit(2)

    # Show configuration
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"⚡ Workers: {workers}")
    print(f"📝 Log Level: {log_level}")
    print(f"🔄 Reload: {reload}")

    if host in {"0.0.0.0", "::"} and not cfg.is_development:
        print("⚠️  Binding to all interfaces. Ensure proper firewall / reverse proxy setup.")

    # API key reminders
    print("\n🔑 Data provider credentials:")
    print("   Alpha Vantage: Set ALPHA_VANTAGE_API_KEY environment variable")
    print("   Polygon.io: Set POLYGON_API_KEY environment variable")
    print("   Alpaca: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
    print("   Tiingo: Set TIINGO_API_KEY environment variable")
    print("   Yahoo Finance: No API key required (free tier)")

    try:
        uvicorn.run(
            "mra_web.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down API server...")
        sys.exit(0)


if __name__ == "__main__":
    main()
