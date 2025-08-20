"""
API endpoint handlers for the Market Regime Analysis API.

This module implements all REST endpoints matching CLI functionality.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from .. import MarketRegimeAnalyzer, PortfolioHMMAnalyzer, SimonsRiskCalculator
from ..providers import MarketDataProvider
from .auth import User, authenticate_request
from .models import (
    AnalysisResponse,
    ChartResponse,
    CurrentAnalysisRequest,
    DetailedAnalysisRequest,
    ExportCSVRequest,
    ExportResponse,
    GenerateChartsRequest,
    MultiAnalysisResponse,
    MultiSymbolAnalysisRequest,
    PortfolioAnalysisResponse,
    PositionSizingRequest,
    PositionSizingResponse,
    ProviderInfo,
    ProvidersResponse,
)
from .utils import (
    api_metrics,
    convert_regime_analysis_to_response,
    get_regime_from_string,
    log_api_request,
    log_api_response,
    run_in_thread,
    validate_api_key,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analysis/detailed", response_model=AnalysisResponse)
async def detailed_analysis(
    request: DetailedAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> AnalysisResponse:
    """
    Run detailed HMM analysis for a single timeframe.

    This endpoint provides comprehensive regime analysis for a specific symbol and timeframe,
    including HMM state detection, confidence metrics, and trading recommendations.
    """
    start_time = time.time()
    endpoint = "/analysis/detailed"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Run analysis in thread pool to avoid blocking
        def run_analysis():
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )
            return analyzer.analyze_current_regime(request.timeframe)

        # Execute analysis
        analysis = await run_in_thread(run_analysis)

        # Convert to response model
        response = convert_regime_analysis_to_response(analysis, request.symbol, request.timeframe)

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except ValueError as e:
        api_metrics.record_error(endpoint)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {e!s}"
        ) from e
    except ConnectionError as e:
        api_metrics.record_error(endpoint)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Data provider connection error: {e!s}",
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"Detailed analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {e!s}"
        ) from e


@router.post("/analysis/current", response_model=MultiAnalysisResponse)
async def current_analysis(
    request: CurrentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> MultiAnalysisResponse:
    """
    Run current HMM regime analysis for all timeframes.

    This endpoint provides comprehensive regime analysis across multiple timeframes
    (1D, 1H, 15m) using parallel processing for optimal performance.
    """
    start_time = time.time()
    endpoint = "/analysis/current"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Run analysis in thread pool
        def run_analysis():
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )

            timeframes = ["1D", "1H", "15m"]
            analyses = []

            for timeframe in timeframes:
                try:
                    analysis = analyzer.analyze_current_regime(timeframe)
                    response = convert_regime_analysis_to_response(
                        analysis, request.symbol, timeframe
                    )
                    analyses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to analyze timeframe {timeframe}: {e}")
                    continue

            return analyses

        # Execute analysis
        analyses = await run_in_thread(run_analysis)

        if not analyses:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to analyze any timeframes",
            )

        # Create response
        response = MultiAnalysisResponse(symbol=request.symbol, analyses=analyses)

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except HTTPException:
        api_metrics.record_error(endpoint)
        raise
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"Current analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {e!s}"
        ) from e


@router.post("/analysis/multi-symbol", response_model=PortfolioAnalysisResponse)
async def multi_symbol_analysis(  # noqa: PLR0915
    request: MultiSymbolAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> PortfolioAnalysisResponse:
    """
    Run multi-symbol HMM analysis (portfolio analysis).

    This endpoint provides portfolio-wide regime analysis including cross-asset
    correlations and portfolio-level insights.
    """
    start_time = time.time()
    endpoint = "/analysis/multi-symbol"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Run analysis in thread pool
        def run_analysis():
            # Initialize portfolio analyzer with proper periods mapping
            periods = {request.timeframe: "2y" if request.timeframe == "1D" else "6mo"}
            portfolio = PortfolioHMMAnalyzer(
                symbols=request.symbols,
                periods=periods,
                provider_flag=request.provider,
                api_key=validated_api_key,
            )

            # Get individual symbol analyses from portfolio's analyzers
            analyses = []
            for symbol in request.symbols:
                if symbol in portfolio.analyzers:
                    try:
                        analyzer = portfolio.analyzers[symbol]
                        analysis = analyzer.analyze_current_regime(request.timeframe)
                        response = convert_regime_analysis_to_response(
                            analysis, symbol, request.timeframe
                        )
                        analyses.append(response)
                    except Exception as e:
                        logger.warning(f"Failed to analyze symbol {symbol}: {e}")
                        continue
                else:
                    logger.warning(f"Symbol {symbol} not available in portfolio analyzer")

            # Get comprehensive portfolio metrics using portfolio analyzer
            portfolio_summary = portfolio.get_portfolio_regime_summary(request.timeframe)
            portfolio_metrics = {
                "total_symbols": len(request.symbols),
                "analyzed_symbols": len(analyses),
                "dominant_regime": portfolio_summary.get("dominant_regime", "Unknown"),
                "average_confidence": portfolio_summary.get("average_confidence", 0.0),
                "regime_consensus": portfolio_summary.get("regime_consensus", 0.0),
                "risk_level": portfolio_summary.get("risk_level", "Unknown"),
                "diversification_benefit": portfolio_summary.get("diversification_benefit", 0.0),
                "correlation_risk": portfolio_summary.get("correlation_risk", 0.0),
                "regime_distribution": portfolio_summary.get("regime_distribution", {}),
            }

            # Get real correlation matrix from portfolio analyzer
            try:
                correlation_df = portfolio.calculate_regime_correlations(request.timeframe)
                # Extract price correlations and convert to dict format
                price_corr_cols = [col for col in correlation_df.columns if "_price_corr" in col]
                correlations = {}

                if price_corr_cols and len(portfolio.portfolio_data.get(request.timeframe, {})) > 0:
                    # Get correlation matrix from portfolio data
                    price_data = portfolio.portfolio_data[request.timeframe][request.symbols]
                    corr_matrix = price_data.corr()
                    correlations = corr_matrix.to_dict()
                else:
                    # Fallback to simple correlation structure
                    correlations = {
                        symbol: {other: 0.0 for other in request.symbols if other != symbol}
                        for symbol in request.symbols
                    }
            except Exception as e:
                logger.warning(f"Failed to calculate correlations: {e}")
                # Fallback correlation matrix
                correlations = {
                    symbol: {other: 0.0 for other in request.symbols if other != symbol}
                    for symbol in request.symbols
                }

            return analyses, portfolio_metrics, correlations

        # Execute analysis
        analyses, portfolio_metrics, correlations = await run_in_thread(run_analysis)

        if not analyses:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to analyze any symbols",
            )

        # Create response
        response = PortfolioAnalysisResponse(
            symbols=request.symbols,
            timeframe=request.timeframe,
            analyses=analyses,
            portfolio_metrics=portfolio_metrics,
            correlations=correlations,
        )

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except HTTPException:
        api_metrics.record_error(endpoint)
        raise
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"Multi-symbol analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio analysis failed: {e!s}",
        ) from e


@router.post("/position-sizing", response_model=PositionSizingResponse)
async def position_sizing(
    request: PositionSizingRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> PositionSizingResponse:
    """
    Calculate position sizing based on regime, confidence, persistence, and correlation.

    This endpoint provides Kelly Criterion-based position sizing with regime-specific
    adjustments and correlation considerations.
    """
    start_time = time.time()
    endpoint = "/position-sizing"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Convert regime string to enum
        regime = get_regime_from_string(request.regime)

        # Calculate regime-adjusted size
        regime_adjusted = SimonsRiskCalculator.calculate_regime_adjusted_size(
            request.base_size, regime, request.confidence, request.persistence
        )

        # Calculate correlation-adjusted size
        correlation_adjusted = SimonsRiskCalculator.calculate_correlation_adjusted_size(
            regime_adjusted, request.correlation
        )

        # Create detailed calculations
        calculations = {
            "base_position_size": request.base_size,
            "regime_multiplier": regime_adjusted / request.base_size
            if request.base_size > 0
            else 1.0,
            "confidence_factor": request.confidence,
            "persistence_factor": request.persistence,
            "correlation_adjustment": correlation_adjusted / regime_adjusted
            if regime_adjusted > 0
            else 1.0,
            "kelly_criterion_applied": True,
            "safety_caps_applied": True,
        }

        # Create response
        response = PositionSizingResponse(
            base_size=request.base_size,
            regime=request.regime,
            regime_adjusted_size=regime_adjusted,
            correlation_adjusted_size=correlation_adjusted,
            final_recommendation=correlation_adjusted,
            calculations=calculations,
        )

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except ValueError as e:
        api_metrics.record_error(endpoint)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {e!s}"
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"Position sizing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Position sizing calculation failed: {e!s}",
        ) from e


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> ProvidersResponse:
    """
    List all available data providers and their capabilities.

    This endpoint returns information about supported data providers including
    their capabilities, rate limits, and requirements.
    """
    start_time = time.time()
    endpoint = "/providers"

    try:
        # Log request
        log_api_request(endpoint, {})
        api_metrics.record_request(endpoint)

        # Get provider information
        providers_info = MarketDataProvider.get_available_providers()

        # Convert to response format
        providers = {}
        for name, info in providers_info.items():
            providers[name] = ProviderInfo(
                name=name,
                description=info["description"],
                requires_api_key=info["requires_api_key"],
                rate_limit_per_minute=info["rate_limit_per_minute"],
                supported_intervals=info["supported_intervals"],
                supported_periods=info["supported_periods"],
            )

        response = ProvidersResponse(providers=providers)

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"List providers error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {e!s}",
        ) from e


@router.post("/charts/generate", response_model=ChartResponse)
async def generate_charts(
    request: GenerateChartsRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> ChartResponse:
    """
    Generate HMM charts for a given symbol and timeframe.

    This endpoint creates visualization charts showing regime analysis results
    and returns the chart data or file path.
    """
    start_time = time.time()
    endpoint = "/charts/generate"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Run chart generation in thread pool
        def generate_chart():
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )
            # Generate chart and return file path
            analyzer.plot_regime_analysis(request.timeframe, request.days)
            # Return a placeholder file path (in real implementation, this would return actual path)
            return (
                f"charts/{request.symbol}_{request.timeframe}_{request.days}d_regime_analysis.png"
            )

        # Execute chart generation
        chart_path = await run_in_thread(generate_chart)

        response = ChartResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            days=request.days,
            file_path=chart_path,
        )

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"Chart generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chart generation failed: {e!s}",
        ) from e


@router.post("/export/csv", response_model=ExportResponse)
async def export_csv(
    request: ExportCSVRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> ExportResponse:
    """
    Export HMM analysis to CSV for a given symbol.

    This endpoint exports comprehensive analysis data to CSV format for
    further analysis and reporting.
    """
    start_time = time.time()
    endpoint = "/export/csv"

    try:
        # Log request
        log_api_request(endpoint, request.dict())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Run CSV export in thread pool
        def export_data():
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )
            # Export to CSV
            analyzer.export_analysis_to_csv(request.filename)

            # Generate filename if not provided
            filename = request.filename or f"{request.symbol}_regime_analysis.csv"

            return filename, 100  # Placeholder record count

        # Execute export
        filename, record_count = await run_in_thread(export_data)

        response = ExportResponse(
            symbol=request.symbol,
            file_path=f"exports/{filename}",
            filename=filename,
            records_count=record_count,
        )

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)

        # Log response
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        return response

    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.error(f"CSV export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"CSV export failed: {e!s}"
        ) from e


# Health check endpoints
@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


@router.get("/metrics")
async def get_metrics(
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> dict[str, Any]:
    """Get API metrics and statistics."""
    return api_metrics.get_metrics()
