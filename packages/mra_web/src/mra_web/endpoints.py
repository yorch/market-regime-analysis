"""
API endpoint handlers for the Market Regime Analysis API.

This module implements all REST endpoints matching CLI functionality. Blocking
analysis work runs in Starlette's thread pool with a timeout (``API_TIMEOUT``).
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from pydantic import ValidationError as PydanticValidationError

from mra_lib import MarketRegimeAnalyzer, PortfolioHMMAnalyzer, SimonsRiskCalculator
from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime
from mra_lib.config.timeframes import TIMEFRAMES
from mra_lib.data_providers import (
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    RateLimitError,
)

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
    periods_for,
    run_in_thread,
    to_jsonable,
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


PROVIDER_RETRY_AFTER_SECONDS = 60


def _exception_chain(exc: BaseException, limit: int = 8) -> list[BaseException]:
    """``exc`` followed by its causes/contexts (the analyzer wraps provider errors)."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < limit:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_exception(exc: Exception) -> HTTPException:  # noqa: PLR0911
    """Map an analysis exception to an HTTP error with a generic, safe detail.

    Provider errors are found anywhere in the cause chain, since
    ``MarketRegimeAnalyzer`` re-raises data-loading failures as ``ValueError``.
    """
    if isinstance(exc, PydanticValidationError):
        # Server-side model failure (a ValueError subclass), not bad client input.
        return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)
    chain = _exception_chain(exc)
    if any(isinstance(e, InvalidSymbolError) for e in chain):
        return HTTPException(status.HTTP_400_BAD_REQUEST, detail="Unknown or invalid symbol")
    if any(isinstance(e, AuthError) for e in chain):
        # Upstream rejected the server's provider credentials: a server-side problem.
        return HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="Data provider authentication failed"
        )
    if any(isinstance(e, RateLimitError) for e in chain):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data provider rate limit reached",
            headers={"Retry-After": str(PROVIDER_RETRY_AFTER_SECONDS)},
        )
    if any(isinstance(e, ConnectionError | TimeoutError) for e in chain):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=PROVIDER_UNAVAILABLE_DETAIL
        )
    if isinstance(exc, ValueError):
        return HTTPException(status.HTTP_400_BAD_REQUEST, detail=INVALID_INPUT_DETAIL)
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)


@asynccontextmanager
async def _tracked(
    endpoint: str,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] | None = None,
    failure_detail: str = "Request failed",
) -> AsyncIterator[None]:
    """Record metrics/logs for an endpoint and map exceptions to HTTP errors.

    ``HTTPException`` passes through unchanged; other exceptions are mapped by
    :func:`classify_exception` (e.g. ``ValueError`` -> 400, ``ConnectionError`` -> 503,
    anything else -> 500 with ``failure_detail``), always with generic details.
    """
    start_time = time.time()
    log_api_request(endpoint, payload or {})
    api_metrics.record_request(endpoint)
    try:
        yield
    except HTTPException:
        api_metrics.record_error(endpoint)
        raise
    except Exception as e:
        api_metrics.record_error(endpoint)
        http_error = classify_exception(e)
        if http_error.status_code >= 500:
            logger.exception("%s failed", endpoint)
        else:
            logger.info("%s rejected input", endpoint, exc_info=True)
        if http_error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            http_error.detail = failure_detail
        raise http_error from e
    else:
        response_time = time.time() - start_time
        api_metrics.record_response_time(endpoint, response_time)
        background_tasks.add_task(log_api_response, endpoint, 200, response_time)


def _analyzer(symbol: str, provider: str, api_key: str, *timeframes: str) -> MarketRegimeAnalyzer:
    """Build an analyzer that loads only the requested timeframes."""
    return MarketRegimeAnalyzer(
        symbol=symbol,
        periods=periods_for(*timeframes),
        provider_flag=provider,
        api_key=api_key,
    )


@router.post("/analysis/detailed", response_model=AnalysisResponse)
async def detailed_analysis(
    request: DetailedAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> AnalysisResponse:
    """
    Run detailed HMM analysis for a single timeframe.

    Only the requested timeframe is loaded from the provider.
    """
    async with _tracked(
        "/analysis/detailed", background_tasks, request.model_dump(), "Analysis failed"
    ):
        validated_api_key = validate_api_key(request.provider, request.api_key)

        def run_analysis() -> RegimeAnalysis:
            analyzer = _analyzer(
                request.symbol, request.provider, validated_api_key, request.timeframe
            )
            return analyzer.analyze_current_regime(request.timeframe)

        analysis = await run_in_thread(run_analysis)
        return convert_regime_analysis_to_response(analysis, request.symbol, request.timeframe)


@router.post("/analysis/current", response_model=MultiAnalysisResponse)
async def current_analysis(
    request: CurrentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> MultiAnalysisResponse:
    """
    Run current HMM regime analysis for all timeframes (1D, 1H, 15m).

    Each timeframe is loaded and analyzed independently, so one unavailable
    timeframe does not fail the others.
    """
    async with _tracked(
        "/analysis/current", background_tasks, request.model_dump(), "Analysis failed"
    ):
        validated_api_key = validate_api_key(request.provider, request.api_key)

        def run_analysis() -> list[AnalysisResponse]:
            analyses = []
            last_error: Exception | None = None
            for timeframe in TIMEFRAMES:
                try:
                    analyzer = _analyzer(
                        request.symbol, request.provider, validated_api_key, timeframe
                    )
                    analysis = analyzer.analyze_current_regime(timeframe)
                    analyses.append(
                        convert_regime_analysis_to_response(analysis, request.symbol, timeframe)
                    )
                except Exception as e:
                    last_error = e
                    logger.warning("Failed to analyze timeframe %s", timeframe, exc_info=True)
            if not analyses and last_error is not None:
                raise last_error  # every timeframe failed: report the real cause
            return analyses

        analyses = await run_in_thread(run_analysis)
        return MultiAnalysisResponse(symbol=request.symbol, analyses=analyses)


def correlations_to_dict(
    corr: pd.DataFrame, symbols: list[str]
) -> dict[str, dict[str, float | None]]:
    """Convert a correlation matrix to a nested mapping; undefined values become None."""
    result: dict[str, dict[str, float | None]] = {}
    for s in symbols:
        row: dict[str, float | None] = {}
        for o in symbols:
            value = corr.at[s, o] if s in corr.index and o in corr.columns else np.nan
            row[o] = float(value) if np.isfinite(value) else None
        result[s] = row
    return result


@router.post("/analysis/multi-symbol", response_model=PortfolioAnalysisResponse)
async def multi_symbol_analysis(
    request: MultiSymbolAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> PortfolioAnalysisResponse:
    """
    Run multi-symbol HMM analysis (portfolio analysis).

    Only the requested timeframe is loaded, each symbol is analyzed once, and
    correlations are computed on returns over the symbols that loaded.
    """
    async with _tracked(
        "/analysis/multi-symbol",
        background_tasks,
        request.model_dump(),
        "Portfolio analysis failed",
    ):
        validated_api_key = validate_api_key(request.provider, request.api_key)
        timeframe = request.timeframe

        def run_analysis() -> tuple[
            dict[str, RegimeAnalysis], dict[str, Any], dict[str, dict[str, float | None]]
        ]:
            portfolio = PortfolioHMMAnalyzer(
                symbols=request.symbols,
                periods=periods_for(timeframe),
                provider_flag=request.provider,
                api_key=validated_api_key,
            )
            analyses = portfolio.collect_analyses(timeframe)
            summary = portfolio.get_portfolio_regime_summary(timeframe, analyses=analyses)
            symbols = list(analyses)
            corr = (
                portfolio.get_return_correlation_matrix(timeframe, symbols)
                if len(symbols) > 1
                else pd.DataFrame()
            )
            return analyses, summary, correlations_to_dict(corr, symbols)

        analyses, summary, correlations = await run_in_thread(run_analysis)

        if not analyses:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to analyze any symbols",
            )

        portfolio_metrics = {
            "total_symbols": len(request.symbols),
            "analyzed_symbols": len(analyses),
            "dominant_regime": summary.get("dominant_regime") or MarketRegime.UNKNOWN.value,
            "average_confidence": summary.get("average_confidence"),
            "regime_consensus": summary.get("regime_consensus", 0.0),
            "risk_level": summary.get("risk_level", "Unknown"),
            # Undefined with fewer than two analyzed symbols.
            "correlation_risk": summary.get("correlation_risk") if len(analyses) > 1 else None,
            "diversification_benefit": summary.get("diversification_benefit")
            if len(analyses) > 1
            else None,
            "regime_distribution": summary.get("regime_distribution", {}),
        }

        return PortfolioAnalysisResponse(
            symbols=request.symbols,
            timeframe=timeframe,
            analyses=[
                convert_regime_analysis_to_response(a, symbol, timeframe)
                for symbol, a in analyses.items()
            ],
            portfolio_metrics=to_jsonable(portfolio_metrics),
            correlations=correlations,
        )


@router.post("/position-sizing", response_model=PositionSizingResponse)
async def position_sizing(
    request: PositionSizingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> PositionSizingResponse:
    """
    Calculate position sizing based on regime, confidence, persistence, and correlation.

    Applies the regime multiplier scaled by confidence and persistence, then a
    correlation adjustment.
    """
    async with _tracked(
        "/position-sizing",
        background_tasks,
        request.model_dump(),
        "Position sizing calculation failed",
    ):
        regime = get_regime_from_string(request.regime)

        regime_adjusted = SimonsRiskCalculator.calculate_regime_adjusted_size(
            request.base_size, regime, request.confidence, request.persistence
        )
        correlation_adjusted = SimonsRiskCalculator.calculate_correlation_adjusted_size(
            regime_adjusted, request.correlation
        )

        calculations = {
            "base_position_size": request.base_size,
            "regime_multiplier": regime_adjusted / request.base_size
            if request.base_size > 0
            else None,
            "confidence_factor": request.confidence,
            "persistence_factor": request.persistence,
            "correlation_adjustment": correlation_adjusted / regime_adjusted
            if regime_adjusted > 0
            else None,
        }

        return PositionSizingResponse(
            base_size=request.base_size,
            regime=request.regime,
            regime_adjusted_size=regime_adjusted,
            correlation_adjusted_size=correlation_adjusted,
            final_recommendation=correlation_adjusted,
            calculations=to_jsonable(calculations),
        )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> ProvidersResponse:
    """
    List all available data providers and their capabilities.

    This endpoint returns information about supported data providers including
    their capabilities, rate limits, and requirements.
    """
    async with _tracked("/providers", background_tasks, None, "Failed to list providers"):
        providers = {
            name: ProviderInfo(
                name=name,
                description=info["description"],
                requires_api_key=info["requires_api_key"],
                rate_limit_per_minute=info["rate_limit_per_minute"],
                supported_intervals=info["supported_intervals"],
                supported_periods=info["supported_periods"],
            )
            for name, info in MarketDataProvider.get_available_providers().items()
        }
        return ProvidersResponse(providers=providers)


@router.post(
    "/charts/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "PNG chart"}},
)
async def generate_charts(
    request: GenerateChartsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> Response:
    """
    Generate the 5-panel HMM regime chart for a symbol and timeframe.

    The chart is rendered in memory with a headless backend and returned as
    ``image/png``; nothing is written on the server.
    """
    async with _tracked(
        "/charts/generate", background_tasks, request.model_dump(), "Chart generation failed"
    ):
        validated_api_key = validate_api_key(request.provider, request.api_key)

        def generate_chart() -> bytes:
            analyzer = _analyzer(
                request.symbol, request.provider, validated_api_key, request.timeframe
            )
            return analyzer.render_regime_chart_png(request.timeframe, request.days)

        png = await run_in_thread(generate_chart)

        filename = f"{request.symbol}_{request.timeframe}_{request.days}_regime_chart.png"
        return Response(
            content=png,
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )


@router.post(
    "/export/csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "CSV export"}},
)
async def export_csv(
    request: ExportCSVRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> Response:
    """
    Export the HMM analysis for a symbol as CSV (one row per available timeframe).

    The CSV is built in memory and returned as ``text/csv``. ``filename`` only sets
    the download name in ``Content-Disposition``; nothing is written on the server.
    The number of data rows is returned in the ``X-Record-Count`` header.
    """
    async with _tracked("/export/csv", background_tasks, request.model_dump(), "CSV export failed"):
        validated_api_key = validate_api_key(request.provider, request.api_key)

        def export_data() -> tuple[str, int]:
            frames = []
            last_error: Exception | None = None
            for timeframe in TIMEFRAMES:
                try:
                    analyzer = _analyzer(
                        request.symbol, request.provider, validated_api_key, timeframe
                    )
                    frame = analyzer.build_export_dataframe()
                    if not frame.empty:
                        frames.append(frame)
                except Exception as e:
                    last_error = e
                    logger.warning("CSV export skipped timeframe %s", timeframe, exc_info=True)
            if not frames and last_error is not None:
                raise last_error  # every timeframe failed: report the real cause
            export_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            return export_df.to_csv(index=False), len(export_df)

        csv_text, record_count = await run_in_thread(export_data)

        if record_count == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No analysis data could be produced for this symbol",
            )

        filename = request.filename or f"{request.symbol}_regime_analysis.csv"
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Record-Count": str(record_count),
            },
        )


# Health check endpoints
@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


@router.get("/metrics")
async def get_metrics(
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> dict[str, Any]:
    """Get API metrics and statistics."""
    return api_metrics.get_metrics()
