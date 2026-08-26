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
uv run uvicorn skyjo.api:app --reload   # dev server at http://localhost:8000
uv run pytest                              # tests
uv run ruff check .                        # lint
```

### Everything, locally deployed
```sh
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/health

## Developer tools

- **MCTS Tree Explorer** (`/tools/mcts-tree`, e.g. http://localhost:5173/tools/mcts-tree) — loads a JSON export from
  `backend/scripts/dump_mcts_tree.py` and lets you browse it interactively: visit shares, priors, Q/U/PUCT, a
  scrubber across simulation-count snapshots, "compare against" deltas, and a move-preference-over-time chart. Not
  linked from the game UI — it's a standalone route for inspecting `mcts_bot` search trees, source in
  `frontend/src/tools/mcts-tree/`.
