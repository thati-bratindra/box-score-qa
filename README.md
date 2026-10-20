# Box Score Q&A

A retrieval-augmented question answering pipeline over a basketball season. Ask it something like *"Which Lanterns starter scored the most total points in games the Lanterns lost by 10 or more points?"* and it answers from the tables, cites every row it used, and abstains when the data can't support an answer.

It runs entirely locally: PostgreSQL with `pgvector` for storage and retrieval, and two small [Ollama](https://ollama.com) models, `nomic-embed-text` for embeddings and `qwen3.5:2b` for planning and reading text.

## How it works

The language model never writes code or computes an answer. Its job is limited to filling in a small, validated plan, and everything else is deterministic Python.

1. **Resolve** (`backend/resolve.py`): find the teams, players, dates and phrases a question names in the tables. Full names win over surnames, near-duplicate names are kept apart, and a surname shared by several players is only accepted when exactly one of them appears in the games the question is about.
2. **Plan** (`backend/plan.py`): build a JSON plan from a fixed set of operations (filter games, look up, sum, count, max/min, streaks, by date range, and so on). Thresholds, outcomes, ordinals and venues are read straight from the wording; the model fills in only what is left, and a plan that doesn't validate falls back to the deterministic skeleton.
3. **Execute** (`backend/execute.py`, `backend/query.py`): run the plan against the tables with stable ordering and fixed tie-breaks, keeping every row that contributed.
4. **Extract and verify** (`backend/extract.py`, `backend/validate.py`): facts that exist only in prose (comeback deficits, largest leads, go-ahead baskets, injury reasons) are pulled from the recaps and injury notes with patterns, checked against the tables, and attributed to the right team.
5. **Answer or abstain** (`backend/rag.py`): every declared field must be filled and every declared evidence table cited, or the question abstains. Questions the plan language can't express fall back to vector retrieval over embedded rows, with the model only allowed to cite rows it was shown.

`backend/server.py` exposes the same path as a `/api/chat` endpoint for the Angular chat UI in `frontend/`, which shows each answer with its evidence rows.

## Data

`backend/data/` holds a **fictional season** with six tables: `teams`, `players`, `game_details`, `player_box_scores`, `game_recaps` (a written recap of every game) and `injury_notes` (pre-game availability reports). There are 16 invented teams, 240 players and 480 games from October 2025 to April 2026, with no games during the All-Star break (February 13-18).

The season is built to be hard on purpose:

* near-duplicate names on the same team (Darian and Dorian Hadderley), shared surnames across and within rosters, and hyphenated surnames that must not match their prefix (Castell-Rooney vs. Castell);
* mid-season trades, so some players appear in box scores for two teams;
* recaps that phrase deficits, leads and go-ahead plays many different ways, and a few recaps that contradict the tables, where the tables must win.

Rules the answers follow: the tables are authoritative over the text, team names are given as mascots, scores as `"<winner points>-<loser points>"`, and every source row used is cited (`game_details` by `game_id`, `player_box_scores` by `game_id` + `person_id`, `game_recaps` by `recap_id`, `injury_notes` by `note_id`).

## Run it

You need [Docker](https://www.docker.com/get-started/) and [Ollama](https://ollama.com/download). Ollama runs natively so it can use your GPU; the app reaches it at `host.docker.internal:11434`.

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:2b

docker compose up -d db
docker compose build app

docker compose run --rm app python -m backend.ingest   # load the CSVs into Postgres
docker compose run --rm app python -m backend.embed    # embed rows into pgvector
docker compose run --rm app python -m backend.rag      # answer questions/questions.json
docker compose run --rm app python -m backend.validate # check questions/answers.json
```

`backend.rag` reads `questions/questions.json` and writes `questions/answers.json`. Each question carries a `return` block that sets the shape of its answer; `questions/answers_template.json` shows that shape and `backend.validate` checks the output against `questions/answers.schema.json`. Ingest replaces the tables on every run, and `docker compose down -v` wipes the database, embeddings included. To smoke-test embedding, run `python -m backend.embed --limit 50`.

### Chat UI

```bash
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

Then in `frontend/` (Node 18, 20 or 22): `npm install && npm start`, and open `http://localhost:4200/`.

### If Ollama can't run natively

```bash
docker compose --profile docker-ollama up -d db ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull qwen3.5:2b
OLLAMA_HOST=http://ollama:11434 docker compose --profile docker-ollama build app
```

Use `OLLAMA_HOST=http://ollama:11434` and the `docker-ollama` profile for every later command.

## Reproducibility

Runs are deterministic: `backend/utils.py` sends `"think": false`, temperature 0 and a fixed seed, retrieval is an exact scan with a stable `ORDER BY` (no approximate index), and every aggregate breaks ties by a fixed rule. The pipeline was built and tested with these model digests:

* `nomic-embed-text`: `0a109f422b47`
* `qwen3.5:2b`: `324d162be6ca`

## More

* [`docs/tracking-data.md`](docs/tracking-data.md): how I would extend the pipeline with player tracking data, and where the current design would break.
