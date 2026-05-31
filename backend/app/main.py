from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
from db.seeds.elements_seed import seed_elements
from db.seeds.yolo_classes_seed import seed_yolo_classes
from db.session import engine
from db.base import Base
from api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await seed_yolo_classes()
    await seed_elements()  # make sure to import this function from the correct module

    yield

    # shutdown (optional)
    # e.g. close connections, cleanup

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )