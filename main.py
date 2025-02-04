from fastapi import APIRouter, FastAPI, Depends, HTTPException, status
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
import crud, schemas, models
from database.db_session import SessionLocal, engine
from database.setup import check_and_create_database
from database.base_class import APIBase
from fastapi.responses import JSONResponse
import logging
from core import settings
from util.auth import route
from apis import router



def create_tables():
    #models.Base.metadata.create_all(bind=engine)
    APIBase.metadata.create_all(bind=engine)

def endpoints(app):
    app.include_router(route)
    app.include_router(router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await check_and_create_database()
        yield
    finally:
        print("Startup completed!")

def start_app():
    app = FastAPI(docs_url="/", title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)
    # CORS middleware
    app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    create_tables()
    endpoints(app)
    lifespan(app)
    return app

app = start_app()

@app.on_event("startup")
async def startup_event():
    async with lifespan(app):
        pass