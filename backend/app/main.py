# Run with: uvicorn app.main:app --reload

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .routes import upload, query
from .config import setup_environment, FRONTEND_PATH

app = FastAPI()

# Add CORS middleware to allow front-end communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup environment variables
setup_environment()

# Include the API routers
app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")

# Mount the front-end static files
app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
