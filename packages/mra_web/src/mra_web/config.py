"""
Configuration management for the Market Regime Analysis API server.

This module provides environment-based configuration for development and production
environments, following the plan specifications.
"""

import os

from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API server configuration settings."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    reload: bool = Field(default=False, description="Enable hot reload for development")

    # Authentication settings
    jwt_secret: str = Field(description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute per client")
    rate_limit_burst: int = Field(default=10, description="Burst limit for rate limiting")

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )
    cors_methods: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS methods"
    )
    cors_headers: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS headers"
    )

    # Environment
    environment: str = Field(default="development", description="Environment (dev/prod)")
    debug: bool = Field(default=False, description="Debug mode")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            timeout=int(os.getenv("API_TIMEOUT", "300")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            jwt_secret=os.getenv("JWT_SECRET", "your-secret-key-change-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "10")),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            cors_methods=os.getenv("CORS_METHODS", "*").split(","),
            cors_headers=os.getenv("CORS_HEADERS", "*").split(","),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global configuration instance
config = APIConfig.from_env()
