"""
API endpoint handlers for the Market Regime Analysis API.

This module implements all REST endpoints matching CLI functionality.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status

from mra_lib import MarketRegimeAnalyzer, PortfolioHMMAnalyzer, SimonsRiskCalculator
from mra_lib.data_providers import MarketDataProvider

from .auth import User, authenticate_request
from .models import (
    AnalysisResponse,
    CurrentAnalysisRequest,
    DetailedAnalysisRequest,
    ExportCSVRequest,
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

# Generic client-facing error details. Exception text is never returned to clients:
# provider errors can embed request URLs that carry API keys.
INVALID_INPUT_DETAIL = "Invalid input or no data available for the requested symbol"
PROVIDER_UNAVAILABLE_DETAIL = "Data provider unavailable"


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
            status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_INPUT_DETAIL
        ) from e
    except ConnectionError as e:
        api_metrics.record_error(endpoint)
        logger.warning("Detailed analysis provider error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=PROVIDER_UNAVAILABLE_DETAIL
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.exception("Detailed analysis error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis failed"
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
                except Exception:
                    logger.warning("Failed to analyze timeframe %s", timeframe, exc_info=True)
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
        logger.exception("Current analysis error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis failed"
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
                    except Exception:
                        logger.warning("Failed to analyze symbol %s", symbol, exc_info=True)
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
            except Exception:
                logger.warning("Failed to calculate correlations", exc_info=True)
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
        logger.exception("Multi-symbol analysis error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
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
            status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_INPUT_DETAIL
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.exception("Position sizing error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Position sizing calculation failed",
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
        logger.exception("List providers error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list providers",
        ) from e


@router.post(
    "/charts/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "PNG chart"}},
)
async def generate_charts(
    request: GenerateChartsRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> Response:
    """
    Generate the 5-panel HMM regime chart for a symbol and timeframe.

    The chart is rendered in memory with a headless backend and returned as
    ``image/png``; nothing is written on the server.
    """
    start_time = time.time()
    endpoint = "/charts/generate"

    try:
        # Log request
        log_api_request(endpoint, request.model_dump())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        # Render chart in thread pool
        def generate_chart() -> bytes:
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )
            return analyzer.render_regime_chart_png(request.timeframe, request.days)

        png = await run_in_thread(generate_chart)

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        filename = f"{request.symbol}_{request.timeframe}_{request.days}_regime_chart.png"
        return Response(
            content=png,
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    except HTTPException:
        api_metrics.record_error(endpoint)
        raise
    except ValueError as e:
        api_metrics.record_error(endpoint)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_INPUT_DETAIL
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.exception("Chart generation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chart generation failed",
        ) from e


@router.post(
    "/export/csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "CSV export"}},
)
async def export_csv(
    request: ExportCSVRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(authenticate_request),  # noqa: B008
) -> Response:
    """
    Export the HMM analysis for a symbol as CSV (one row per timeframe).

    The CSV is built in memory and returned as ``text/csv``. ``filename`` only sets
    the download name in ``Content-Disposition``; nothing is written on the server.
    The number of data rows is returned in the ``X-Record-Count`` header.
    """
    start_time = time.time()
    endpoint = "/export/csv"

    try:
        # Log request
        log_api_request(endpoint, request.model_dump())
        api_metrics.record_request(endpoint)

        # Validate API key
        validated_api_key = validate_api_key(request.provider, request.api_key)

        def export_data() -> tuple[str, int]:
            analyzer = MarketRegimeAnalyzer(
                symbol=request.symbol, provider_flag=request.provider, api_key=validated_api_key
            )
            export_df = analyzer.build_export_dataframe()
            return export_df.to_csv(index=False), len(export_df)

        csv_text, record_count = await run_in_thread(export_data)

        if record_count == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No analysis data could be produced for this symbol",
            )

        # Record metrics
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)

        filename = request.filename or f"{request.symbol}_regime_analysis.csv"
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Record-Count": str(record_count),
            },
        )

    except HTTPException:
        api_metrics.record_error(endpoint)
        raise
    except ValueError as e:
        api_metrics.record_error(endpoint)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_INPUT_DETAIL
        ) from e
    except Exception as e:
        api_metrics.record_error(endpoint)
        logger.exception("CSV export error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="CSV export failed"
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
