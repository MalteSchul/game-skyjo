from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skyjo.api.matches import router as matches_router

app = FastAPI(title="skyjo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
