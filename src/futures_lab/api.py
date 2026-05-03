import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from futures_lab.config import get_settings
from futures_lab.models import Decision, MarketState, PaperState, RiskVerdict
from futures_lab.runtime import TradingRuntime

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

runtime = TradingRuntime.create(settings)

app = FastAPI(title="Futures Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8090", "http://localhost:8090"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not settings.api_token:
        return
    if x_api_key != settings.api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if runtime.is_running():
        await runtime.stop()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "symbol": settings.symbol.upper(),
        "running": runtime.is_running(),
        "env": settings.app_env,
    }


@app.post("/runtime/start")
def start(_: None = Depends(require_api_key)) -> dict:
    runtime.start()
    return {"running": runtime.is_running(), "symbol": settings.symbol.upper()}


@app.post("/runtime/stop")
async def stop(_: None = Depends(require_api_key)) -> dict:
    await runtime.stop()
    return {"running": runtime.is_running(), "symbol": settings.symbol.upper()}


@app.get("/market", response_model=MarketState)
def market(_: None = Depends(require_api_key)) -> MarketState:
    return runtime.market()


@app.get("/decision")
def decision(_: None = Depends(require_api_key)) -> dict:
    market, proposed, risk = runtime.decide_once()
    return {
        "market": market.model_dump(),
        "decision": proposed.model_dump(),
        "risk": risk.model_dump(),
    }


@app.get("/paper", response_model=PaperState)
def paper(_: None = Depends(require_api_key)) -> PaperState:
    return runtime.paper_state()


@app.post("/paper/reset", response_model=PaperState)
def reset_paper(_: None = Depends(require_api_key)) -> PaperState:
    return runtime.paper.reset()


@app.get("/latest")
def latest(_: None = Depends(require_api_key)) -> dict:
    return {
        "market": runtime.latest_market.model_dump() if runtime.latest_market else None,
        "decision": runtime.latest_decision.model_dump() if runtime.latest_decision else None,
        "risk": runtime.latest_risk.model_dump() if runtime.latest_risk else None,
        "paper": runtime.paper_state().model_dump(),
    }

