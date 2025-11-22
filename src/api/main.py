# src/api/main.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

def create_app():
    app = FastAPI()

    @app.on_event("startup")
    async def enable_metrics():
        Instrumentator().instrument(app).expose(app)

    return app