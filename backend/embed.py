"""Build and embed natural-language documents for every retrievable table.

Generates one document per row for game_details, player_box_scores, game_recaps
and injury_notes, joining in names so each document reads like a sentence a human
would write rather than a row of foreign keys, embeds them in batches with
nomic-embed-text, and stores the text and its vector back on the source table.

All four tables run through the same helper (`embed_table`), so adding a fifth
is a matter of describing it rather than copying a pipeline.

What retrieval is *for* here is worth stating, because it shapes the documents.
Most questions are answered by backend/execute.py running a structured query, not
by similarity search -- a season total or a per-game maximum cannot be found by
ranking documents. Embeddings carry the narrative tier: they narrow recaps and
injury notes down to the passage worth reading, and they back the fallback path
in rag.py when a question cannot be planned. The documents are written to be
discriminative for that job: each one leads with the entities (names, teams,
dates) a question is likely to mention.

usage:
    python -m backend.embed [--limit N]

    --limit N   Only embed the first N rows of each table (most recent first),
                for a fast smoke run instead of embedding the whole season.
"""
import argparse

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed_batch

# nomic-embed-text is an asymmetric model: text being indexed gets the
# "search_document: " prefix and text being searched gets "search_query: ".
# Ollama does not add these itself, so each side of retrieval adds its own.
DOCUMENT_PREFIX = "search_document: "

TEAM_LABEL_SQL = "{alias}.city || ' ' || {alias}.name || ' (' || {alias}.abbreviation || ')'"

# ingest.py stores game_timestamp as text, so it is cast before formatting.
GAME_DATE_SQL = "g.game_timestamp::timestamp::date"


def build_game_doc(row: pd.Series) -> str:
    """Build a natural-language document for one game_details row.

    Joins in home/away team city+name+abbreviation (from the teams table, already
    merged onto `row` by the caller) so the document reads like a sentence a
    human would write, rather than raw foreign keys. This is what game retrieval
    in rag.py and server.py searches against.

    :param row: one row of game_details left-joined with teams for home and away
        (expects home_team_label, away_team_label, home_points, away_points,
        winning_team_id, home_team_id, game_timestamp, game_id)
    :return: one-paragraph natural-language description of the game
    """
    date = pd.to_datetime(row.game_timestamp).strftime("%Y-%m-%d")
    home_points = int(row.home_points)
    away_points = int(row.away_points)
    if row.winning_team_id == row.home_team_id:
        winner_label, winner_points = row.home_team_label, home_points
        loser_label, loser_points = row.away_team_label, away_points
    else:
        winner_label, winner_points = row.away_team_label, away_points
        loser_label, loser_points = row.home_team_label, home_points

    return (
        f"On {date}, {row.home_team_label} hosted {row.away_team_label}. "
        f"Final score: {winner_label} {winner_points}, {loser_label} {loser_points}. "
        f"Winner: {winner_label}. (game_id={int(row.game_id)})"
    )


def build_box_doc(row: pd.Series) -> str:
    """Build a document for one player_box_scores row.

    Leads with the player's full name and both teams, since those are what a
    question names. The stat line follows so a similarity search for "who scored
    40" has something to match on.

    :param row: a player_box_scores row joined to players, teams and game_details
    :return: one-sentence description of that player's game
    """
    role = "started" if bool(row.starter) else "came off the bench"
    minutes = int(round(float(row.seconds or 0) / 60))
    return (
        f"On {row.game_date}, {row.player_name} of {row.team_label} {role} "
        f"against {row.opponent_label} and scored {int(row.points)} points "
        f"with {int(row.rebounds)} rebounds, {int(row.assists)} assists, "
        f"{int(row.steals)} steals, {int(row.blocks)} blocks and "
        f"{int(row.turnovers)} turnovers in {minutes} minutes. "
        f"(game_id={int(row.game_id)}, person_id={int(row.person_id)})"
    )


def build_recap_doc(row: pd.Series) -> str:
    """Build a document for one game_recaps row.

    The recap is kept whole rather than chunked: each one is a single paragraph
    describing one game, so splitting it would separate a fact ("the go-ahead
    bucket came from X") from the context that identifies the game. Chunking
    would start to pay off if recaps grew to multiple paragraphs.

    The date and both team labels are prepended because the recap prose itself
    often refers to teams only by city or abbreviation, which makes a bare recap
    hard to retrieve by a question that names the full team.

    :param row: a game_recaps row joined to game_details and teams
    :return: the recap text with a retrieval header
    """
    return (
        f"Game recap for {row.game_date}: {row.home_team_label} vs "
        f"{row.away_team_label}. {row.body} "
        f"(recap_id={int(row.recap_id)}, game_id={int(row.game_id)})"
    )


def build_note_doc(row: pd.Series) -> str:
    """Build a document for one injury_notes row.

    The note text names the player by surname only ("Larrabee is out ... with
    hamstring tightness"), so the player's full name, team and opponent are
    prepended from the joins. Without that, a question naming the player in full
    would not retrieve their own note.

    :param row: an injury_notes row joined to players, teams and game_details
    :return: the note text with an identifying header
    """
    return (
        f"Injury note for {row.player_name} of {row.team_label}, "
        f"{row.note_date}, versus {row.opponent_label}: status {row.status}. "
        f"{row.body} (note_id={int(row.note_id)}, game_id={int(row.game_id)})"
    )


# Each entry describes a table completely enough for embed_table to handle it:
# where the rows come from, how to turn one into prose, and how to write it back.
# No vector index on any of them -- see the note in main().
TABLES = {
    "game_details": {
        "key": "game_id",
        "builder": build_game_doc,
        "query": (
            "SELECT g.game_id, g.game_timestamp, g.home_team_id, g.away_team_id, "
            "g.home_points, g.away_points, g.winning_team_id, "
            f"{TEAM_LABEL_SQL.format(alias='home')} AS home_team_label, "
            f"{TEAM_LABEL_SQL.format(alias='away')} AS away_team_label "
            "FROM game_details g "
            "JOIN teams home ON home.team_id = g.home_team_id "
            "JOIN teams away ON away.team_id = g.away_team_id "
            "ORDER BY g.game_timestamp DESC, g.game_id DESC"
        ),
    },
    "player_box_scores": {
        # Composite key: a box score row is identified by the player and the game.
        "key": ("game_id", "person_id"),
        "builder": build_box_doc,
        "query": (
            "SELECT b.game_id, b.person_id, b.starter, b.seconds, b.points, "
            "b.assists, b.steals, b.blocks, b.turnovers, "
            "b.offensive_reb + b.defensive_reb AS rebounds, "
            f"{GAME_DATE_SQL} AS game_date, "
            "p.first_name || ' ' || p.last_name AS player_name, "
            f"{TEAM_LABEL_SQL.format(alias='t')} AS team_label, "
            f"{TEAM_LABEL_SQL.format(alias='o')} AS opponent_label "
            "FROM player_box_scores b "
            "JOIN game_details g ON g.game_id = b.game_id "
            "JOIN players p ON p.player_id = b.person_id "
            "JOIN teams t ON t.team_id = b.team_id "
            "JOIN teams o ON o.team_id = CASE WHEN g.home_team_id = b.team_id "
            "THEN g.away_team_id ELSE g.home_team_id END "
            "ORDER BY g.game_timestamp DESC, b.game_id DESC, b.person_id DESC"
        ),
    },
    "game_recaps": {
        "key": "recap_id",
        "builder": build_recap_doc,
        "query": (
            'SELECT r.recap_id, r.game_id, r."text" AS body, '
            f"{GAME_DATE_SQL} AS game_date, "
            f"{TEAM_LABEL_SQL.format(alias='home')} AS home_team_label, "
            f"{TEAM_LABEL_SQL.format(alias='away')} AS away_team_label "
            "FROM game_recaps r "
            "JOIN game_details g ON g.game_id = r.game_id "
            "JOIN teams home ON home.team_id = g.home_team_id "
            "JOIN teams away ON away.team_id = g.away_team_id "
            "ORDER BY g.game_timestamp DESC, r.recap_id DESC"
        ),
    },
    "injury_notes": {
        "key": "note_id",
        "builder": build_note_doc,
        "query": (
            'SELECT n.note_id, n.game_id, n.note_date, n.status, n."text" AS body, '
            "p.first_name || ' ' || p.last_name AS player_name, "
            f"{TEAM_LABEL_SQL.format(alias='t')} AS team_label, "
            f"{TEAM_LABEL_SQL.format(alias='o')} AS opponent_label "
            "FROM injury_notes n "
            "JOIN game_details g ON g.game_id = n.game_id "
            "JOIN players p ON p.player_id = n.player_id "
            "JOIN teams t ON t.team_id = n.team_id "
            "JOIN teams o ON o.team_id = CASE WHEN g.home_team_id = n.team_id "
            "THEN g.away_team_id ELSE g.home_team_id END "
            "ORDER BY n.note_date DESC, n.note_id DESC"
        ),
    },
}


def embed_table(cx, table: str, spec: dict, limit: int | None) -> int:
    """Build, embed and store documents for one table.

    :param cx: open SQLAlchemy connection inside a transaction
    :param table: table name to write `doc` and `embedding` back to
    :param spec: the TABLES entry describing this table
    :param limit: only embed this many rows, or None for all
    :return: number of rows updated
    """
    # `doc` holds the human-readable text so it can be inspected/debugged
    # directly in SQL; `embedding` is the vector derived from it.
    cx.execute(text(f"ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS doc text;"))
    cx.execute(text(f"ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS embedding vector(768);"))

    query = spec["query"]
    if limit is not None:
        query += " LIMIT :limit"
        df = pd.read_sql(text(query), cx, params={"limit": limit})
    else:
        df = pd.read_sql(text(query), cx)
    if df.empty:
        print(f"  {table}: no rows")
        return 0

    docs = [spec["builder"](r) for _, r in df.iterrows()]
    vectors = ollama_embed_batch(EMBED_MODEL, [DOCUMENT_PREFIX + d for d in docs])

    keys = spec["key"] if isinstance(spec["key"], tuple) else (spec["key"],)
    where = " AND ".join(f"{k} = :{k}" for k in keys)
    update = text(f"UPDATE {table} SET doc = :doc, embedding = :v WHERE {where}")
    for (_, row), doc, vec in zip(df.iterrows(), docs, vectors):
        params = {"doc": doc, "v": vec}
        params.update({k: int(row[k]) for k in keys})
        cx.execute(update, params)

    print(f"  {table}: {len(df)} rows")
    return len(df)


def main(limit: int | None) -> None:
    """Embed every table's documents and store them with their vectors.

    :param limit: if given, only embed this many rows per table (most recent
        first), for a quick smoke test instead of a full run.
    """
    print("Starting Embedding Process")
    eng = sa.create_engine(DB_DSN)
    total = 0
    with eng.begin() as cx:
        cx.execute(text("ALTER DATABASE nba REFRESH COLLATION VERSION"))
        # No vector index on any table on purpose. game_details has under 1k rows
        # and even player_box_scores is only ~18k, so an exact cosine scan (what
        # rag.py's ORDER BY does without an index) costs milliseconds -- there's
        # no performance reason to add one. An approximate index (hnsw/ivfflat) would be actively
        # harmful for reproducibility: pgvector's hnsw assigns node levels randomly
        # during construction, so a fresh embed builds a differently-shaped
        # graph every time, which can flip which rows tie for the last
        # top-k slot and make retrieve() disagree with the committed
        # questions/answers.json. If you add an index anyway, make sure
        # retrieval still returns results in the same order on every build
        # (e.g. keep an explicit tie-breaker in ORDER BY, as rag.py does).
        for table, spec in TABLES.items():
            total += embed_table(cx, table, spec, limit)
    print(f"Finished Embeddings: {total} Rows Updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents into pgvector.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only embed the first N rows per table, for a quick smoke run.")
    args = parser.parse_args()
    main(args.limit)
