import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from lse import LSE

load_dotenv()
API_KEY: str = os.environ["LSE_API_KEY"] # raises error if missing
SYMBOL: str = "MANU"

# Latest tick kept in memory only (no persistence).
last: dict[str, Any] | None = None
connected: bool = False


async def stream():
    """Pull live LSE ticks into `last`; reconnect after transient errors."""
    global last, connected
    try:
        async for t in LSE(api_key=API_KEY).stream_async([SYMBOL], reconnect=True):
            connected = True
            last = {
                "symbol": getattr(t, "symbol", SYMBOL),
                "price": float(t.price),
                "bid": t.bid,
                "ask": t.ask,
                "volume": getattr(t, "volume", None),
                "received_at": time.time(),
            }
    finally:
        # reconnect happens inside the async loop
        connected = False

@asynccontextmanager
async def lifespan(_app):
    # Start the LSE stream with the API; cancel it on shutdown.
    task = asyncio.create_task(stream())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "connected": connected, "has_tick": last is not None}


@app.get("/tick")
def tick():
    """Last price + bid/ask. 503 until the first tick arrives."""
    if not last:
        raise HTTPException(503, "no tick yet")
    return {**last, "age_ms": int((time.time() - last["received_at"]) * 1000)}

# local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)