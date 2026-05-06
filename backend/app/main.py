from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.session import engine
from db.base import Base
from api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    # shutdown (optional)
    # e.g. close connections, cleanup

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api")