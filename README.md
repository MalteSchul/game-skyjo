# Skyjo

A web implementation of the [Skyjo](https://en.wikipedia.org/wiki/Skyjo) card game, with a reinforcement-learning agent as an opponent.

- **`frontend/`** — React + TypeScript + Vite. Renders the game and calls the backend for AI moves.
- **`backend/`** — Python (uv-managed). The Skyjo rules/environment, RL training code, and a FastAPI service that serves a trained agent's moves to the frontend.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the branching workflow, testing convention, and definition of done.

## Quick start

### Frontend
```sh
cd frontend
npm install
npm run dev       # dev server at http://localhost:5173
npm test          # Vitest
npm run build     # production build
```

### Backend
```sh
cd backend
uv sync
uv run uvicorn skyjo_rl.api:app --reload   # dev server at http://localhost:8000
uv run pytest                              # tests
uv run ruff check .                        # lint
```

### Everything, locally deployed
```sh
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/health
