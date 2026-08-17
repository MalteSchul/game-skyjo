# Contributing

## Branching

Trunk-based: `main` is always green and deployable.

- Branch off `main`, name it `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, or `docs/<slug>`.
- Open a PR into `main`.
- Squash-merge, then delete the branch.
- No direct pushes to `main` — everything goes through a PR (enforced by branch protection).

## Testing convention

Every change needs tests covering three cases:

- **Happy path** — normal, expected input/usage works correctly.
- **Sad path** — invalid or unexpected input is rejected or handled gracefully (no crash, correct error).
- **Bad path** — adversarial or boundary conditions (edge values, empty/extreme state) still behave correctly.

See `frontend/src/game/deck.test.ts` and `backend/tests/test_deck.py` for the pattern in practice.

- Frontend: Vitest (`npm test` in `frontend/`).
- Backend: pytest (`uv run pytest` in `backend/`).

## Definition of done

A piece of work isn't done until all three are true:

1. **Tested** — happy/sad/bad tests added or updated, passing locally and in CI.
2. **Committed** — clear commit message, no `Co-Authored-By: Claude` (or any AI) trailer, on a feature branch merged via PR.
3. **Deployed** — `docker compose up --build` succeeds from a clean checkout and both services are reachable (frontend loads, `GET /health` on the backend responds, frontend successfully calls the backend). This is the deploy target until a real host is chosen.

## Local setup

See the Quick start section in [README.md](./README.md).
