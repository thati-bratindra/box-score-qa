"""Answer questions/questions.json with retrieval-augmented generation.

The pipeline is plan-and-execute rather than retrieve-and-summarize:

    resolve  entities in the question, exactly, against the tables (resolve.py)
    plan     the LLM fills in a small declarative query form   (plan.py)
    execute  the plan runs as SQL; the rows it touches become
             the evidence                                      (execute.py)
    extract  facts that exist only in prose are read out of
             recaps and notes, then verified against the tables (extract.py)
    verify   rule-based abstention
    format   coerce to the question's declared `return` shape

Similarity search over documents is still here and still load-bearing: it
narrows recaps and injury notes, and answer_by_similarity is the fallback when a
question cannot be planned. What it is not is the primary path for a total, a
season maximum or an ordinal -- those are not ranking problems, and no top-k of
documents contains the answer to "how many points in January".

The model never produces a number that reaches an answer. It chooses the shape
of the query and it reads prose; everything else is computed from the tables.

usage:
    python -m backend.rag
"""
import json
import os
import re

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.execute import apply_single_game_values, execute_plan, select_games
from backend.extract import (GO_AHEAD_PRESENT, GO_AHEAD_TRIGGER, game_roster, llm_extract, load_notes,
                             load_recaps, pick_concept, regex_person, regex_reason,
                             resolve_name_in_game, scan_recaps_for_number,
                             split_sentences)
from backend.plan import (apply_inferred, audit_plan, build_prompt,
                          default_outputs, enforce_output_coherence,
                          finalize_outputs, prefill_plan, supported_concept,
                          unrecognized_name_phrase,
                          unsupported_concept, validate_plan)
from backend.resolve import (disambiguate_person, load_players, load_season_bounds,
                             load_teams, resolve_question, unresolved_proper_nouns,
                             unresolved_subject, whole_word_match)
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "questions", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "questions", "answers.json"))

TOP_K = 5
PLAN_ATTEMPTS = 2
# nomic-embed-text is an asymmetric embedding model: text being indexed gets the
# "search_document: " prefix (see embed.py) and text being searched gets
# "search_query: " instead. utils.ollama_embed does not add this itself, so
# both sides of the retrieval must add their own prefix consistently.
QUERY_PREFIX = "search_query: "

# Tables that hold prose rather than rows of facts. A question declaring one is
# declaring that part of its answer is only written down, not tabulated.
TEXT_TABLES = {"game_recaps", "injury_notes"}

# Sentinel distinguishing "the query ran and nothing qualified" from "the
# question could not be turned into a query". The first is an answer -- the data
# does not support the claim -- and must not be retried with similarity search,
# which would happily produce a number anyway. The second is a gap in the plan
# DSL, and is worth a second attempt by other means.
NO_ROWS = object()


def retrieve(cx: Connection, qvec: list[float], k: int = TOP_K, table: str = "game_details",
             id_column: str = "game_id", game_ids: list[int] | None = None) -> list[dict]:
    """Retrieve the k nearest documents to a query embedding.

    Shared by rag.py and server.py so both use the exact same retrieval query
    and column set (see server.py: import from here instead of duplicating).

    `game_ids` applies a metadata filter before ranking, which is how the text
    tier works: the structured filters decide which games are in play, and
    similarity only orders the passages within them. Ranking the whole corpus and
    hoping the right game floats to the top is strictly worse when the game is
    already known.

    :param cx: open SQLAlchemy connection
    :param qvec: query embedding, same dimensionality as the table's embedding
    :param k: number of rows to retrieve
    :param table: which embedded table to search
    :param id_column: that table's primary key column
    :param game_ids: optional metadata pre-filter restricting to these games
    :return: rows with id, doc, and cosine similarity score, best first
    """
    # Casting the bind param to ::vector is required: without it Postgres can't
    # tell psycopg2's array/list parameter apart from pgvector's `vector` type
    # and the query errors instead of comparing embeddings.
    where = ["embedding IS NOT NULL"]
    params: dict = {"q": qvec, "k": k}
    if game_ids:
        where.append("game_id = ANY(:gids)")
        params["gids"] = [int(g) for g in game_ids]
    sql = (
        f"SELECT {id_column} AS id, doc, 1 - (embedding <=> (:q)::vector) AS score "
        f"FROM {table} WHERE " + " AND ".join(where) + " "
        # id as a tie-breaker: rows with equal cosine distance must sort
        # the same way on every run, or a fresh embed can disagree with the
        # committed questions/answers.json on the top-k order.
        f"ORDER BY embedding <=> (:q)::vector, {id_column} LIMIT :k"
    )
    return [dict(r) for r in cx.execute(text(sql), params).mappings().all()]


def build_context(rows: list[dict], table: str = "game_details") -> str:
    """Render retrieved rows as the context block shown to the LLM.

    :param rows: rows as returned by retrieve() (needs id and doc)
    :param table: table name to label each line with
    :return: one "[<table> id=<id>] <doc>" line per row
    """
    return "\n".join(f"[{table} id={r['id']}] {r['doc']}" for r in rows)


def coerce_type(value, type_name: str):
    """Coerce a raw JSON value to the type named in a question's `return` block.

    :param value: value as parsed from the model's JSON output
    :param type_name: one of "bool", "int", "str" (the vocabulary used in
        questions/questions.json's `return` blocks)
    :return: value converted to that type
    :raises ValueError: if type_name is unrecognized
    :raises TypeError: if value can't be coerced to that type
    """
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if type_name == "int":
        return int(round(float(value)))
    if type_name in ("float", "number"):
        # Averages are the reason this exists. Rounded to two places so a
        # recomputed value is byte-identical on a rerun rather than differing in
        # the last bit of a repeating decimal.
        return round(float(value), 2)
    if type_name == "str":
        # A whole-number float rendered as a string should read "23", not
        # "23.0" -- the latter looks like a typo in an answer.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    raise ValueError(f"Unknown type in return schema: {type_name}")


def extract_json(raw: str) -> dict | None:
    """Pull the first top-level JSON object out of an LLM's free-text response.

    Models asked for JSON often wrap it in prose or code fences anyway, so this
    finds the outermost {...} span and parses it rather than requiring the
    whole response to be valid JSON.

    :param raw: raw text returned by ollama_generate
    :return: parsed dict, or None if no valid JSON object could be found
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_evidence(result, declared_shapes: list[dict]) -> list[dict]:
    """Turn the rows the executor touched into the citation list.

    This inverts the usual arrangement. Rather than asking the model to cite its
    sources and then checking whether those rows exist, the executor records the
    rows it read while computing, and those rows *are* the citation. A model can
    no longer pair a real row with a wrong number, because the number was derived
    from that row.

    Only tables the question's `return.evidence` block declares are emitted, and
    each is shaped with the keys that block names -- `game_details` by id,
    `player_box_scores` by game_id + person_id, and so on.

    :param result: an ExecResult after execute_plan() and any text extraction
    :param declared_shapes: question["return"]["evidence"]
    :return: evidence rows, sorted so repeated runs emit identical output
    """
    declared = {s["table"] for s in declared_shapes if "table" in s}
    evidence: list[dict] = []

    if "game_details" in declared:
        evidence += [{"table": "game_details", "id": gid} for gid in sorted(result.games)]
    if "player_box_scores" in declared:
        evidence += [{"table": "player_box_scores", "game_id": g, "person_id": p}
                     for g, p in sorted(result.box)]
    if "game_recaps" in declared:
        evidence += [{"table": "game_recaps", "id": rid} for rid in sorted(result.recaps)]
    if "injury_notes" in declared:
        evidence += [{"table": "injury_notes", "id": nid} for nid in sorted(result.notes)]
    return evidence


def normalize_team_names(result: dict, teams: list[dict]) -> dict:
    """Force team references in string fields down to the mascot name.

    Small local models (qwen3.5:2b here) reliably ignore the prompt's
    "use team mascots only" instruction and keep writing "Sablecrest Outriders (SBC)"
    instead of "Outriders" -- so instead of relying on the prompt, this enforces
    the README's naming contract in code after the model has already answered.

    Only touches string fields, skips "evidence", and skips player_name/reason/status
    since those are free-text/person fields, not team references. A field is only
    rewritten when its whole value contains exactly one team's "City Name",
    mascot, or abbreviation as a whole word -- ambiguous matches (e.g. a value
    that happens to mention two teams) are left alone rather than guessed at.

    :param result: coerced result dict, keyed by the question's `return` fields
    :param teams: teams loaded by load_teams()
    :return: a new dict with team references in string fields normalized to mascot names
    """
    skip_fields = {"evidence", "player_name", "reason", "status"}
    normalized = dict(result)
    for key, value in result.items():
        if key in skip_fields or not isinstance(value, str):
            continue
        matched_mascots = {
            team["name"]
            for team in teams
            if re.search(rf"\b{re.escape(team['city'] + ' ' + team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['abbreviation'])}\b", value)
        }
        if len(matched_mascots) == 1:
            normalized[key] = next(iter(matched_mascots))
    return normalized


def normalize_player_names(result: dict, players: list[dict]) -> dict:
    """Rewrite player references to the canonical "First Last" from players.csv.

    The counterpart to normalize_team_names, and needed for the same reason: the
    README fixes how names must be written, and prose sources write them every
    other way ("Castell-Rooney (PCR)", "Ferreira", "Florian Ferreira"). Enforcing the
    format in code is more reliable than asking a 2B model to remember it.

    A value is only rewritten when it resolves to exactly one player. A bare
    surname shared by several players is left untouched, because choosing one of
    them here would be a guess -- surname disambiguation belongs in
    extract.resolve_name_in_game, where the game roster is available.

    :param result: coerced result dict
    :param players: players loaded by load_players()
    :return: a new dict with player references canonicalized
    """
    name_fields = {"player_name", "winner", "opponent", "reason", "status"}
    normalized = dict(result)
    for key, value in result.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if key not in name_fields or key in ("reason", "status"):
            continue
        exact = [p for p in players if p["full_name"].lower() == value.strip().lower()]
        if len(exact) == 1:
            normalized[key] = exact[0]["full_name"]
            continue
        matches = {p["full_name"] for p in players if whole_word_match(p["full_name"], value)}
        if len(matches) == 1:
            normalized[key] = next(iter(matches))
    return normalized


def abstain_result(return_spec: dict) -> dict:
    """Build a schema-valid abstention result straight from a question's `return` block.

    answers_template.json's defaults are shape placeholders (candidates fill them
    in), not a real answer -- reusing them for an abstention silently shipped
    "answerable": true with id-0 evidence, which reads as a fabricated
    fact rather than a deliberate "I don't know". An abstention must always be
    answerable=false with empty/zeroed fields and no evidence.

    :param return_spec: question["return"], the declared field name -> type map
    :return: a result dict with answerable False, ints 0, strings "", evidence []
    """
    zeros = {"bool": False, "int": 0, "float": 0.0, "number": 0.0, "str": ""}
    result = {key: zeros[type_name] for key, type_name in return_spec.items() if key != "evidence"}
    result["answerable"] = False
    result["evidence"] = []
    return result


def evidence_fully_grounded(declared_shapes: list[dict], grounded: list[dict]) -> bool:
    """Check that every evidence table the question requires has a grounded row.

    A question is only answerable if every table its `return.evidence` block
    lists actually has at least one verified supporting row -- e.g. a question
    that cites both game_details and game_recaps isn't answerable off
    game_details alone.

    :param declared_shapes: question["return"]["evidence"]
    :param grounded: output of build_evidence()
    :return: True if every declared table has at least one grounded row
    """
    declared_tables = {s["table"] for s in declared_shapes}
    grounded_tables = {g["table"] for g in grounded}
    return bool(declared_tables) and declared_tables.issubset(grounded_tables)


def fill_text_fields(cx: Connection, question: dict, plan: dict, result,
                     teams: dict, fields: list[str]) -> dict:
    """Read the return fields that exist only in prose, and verify each one.

    Handles the two shapes the text tier actually takes:

    * **Scan and aggregate.** "The largest deficit overcome this season" is a
      maximum over every candidate game's recap, so all of them are read and the
      extreme wins. No top-k can answer this: similarity ranks passages by how
      much they read like the question, which has nothing to do with which one
      holds the biggest number. Once the winning game is known, the facts around
      it (score, opponent) become answerable from the table and are recomputed.
    * **Pinpoint.** The filters already identify one game, so the one recap or
      note attached to it is read directly.

    Every value is then checked: a number must be stated in the passage, and a
    person must resolve to exactly one player on that game's roster. An
    unverifiable value is dropped, which makes the caller abstain.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param plan: the validated plan
    :param result: ExecResult to record recap/note evidence into
    :param teams: team_id -> team dict
    :param fields: return-block fields the plan mapped to the "text" primitive
    :return: field name -> extracted value, omitting anything unverifiable
    """
    out: dict = {}
    games = select_games(cx, plan["game_filters"], plan["game_select"])
    if not games:
        return out
    subject_id = result.values.get("subject_team")
    subject = teams.get(int(subject_id)) if subject_id is not None else None

    if plan["source"] == "injury_notes":
        return _fill_from_notes(cx, question, plan, result, fields, games)

    # A season-wide numeric scan: the plan asks for an extreme of something only
    # the recaps state.
    numeric_field = next(
        (f for f in fields
         if question["return"].get(f) == "int" and pick_concept(question["question"], f)), None)
    if numeric_field and len(games) > 1:
        concept = pick_concept(question["question"], numeric_field)
        found = scan_recaps_for_number(cx, games, concept, subject,
                                       question["question"], numeric_field)
        if not found:
            return out
        take_max = plan["aggregate"]["op"] != "min"
        # -game_id in the sort key breaks ties toward the lowest game id, so a
        # rerun picks the same game.
        best_gid, (value, recap_id) = (max if take_max else min)(
            found.items(), key=lambda kv: (kv[1][0], -kv[0]))
        out[numeric_field] = value

        # Every recap that supplied a value took part in the comparison, so all
        # of them are cited -- with the game each one describes, which the
        # README asks for explicitly. Recaps that stated no deficit contributed
        # nothing and are left out, the same way a game outside the date range
        # is left out of a monthly total.
        by_game = {int(g["game_id"]): g for g in games}
        result.games.clear()
        for gid, (_value, rid) in sorted(found.items()):
            result.recaps.add(int(rid))
            if gid in by_game:
                result.touch_game(by_game[gid])

        # The score and opponent still describe the one game that won the
        # comparison, not the whole candidate set.
        if int(best_gid) in by_game:
            apply_single_game_values(result, by_game[int(best_gid)], subject_id, teams)
        return out

    return _fill_from_recap(cx, question, result, fields, games, subject,
                            numeric_field, teams)


def _fill_from_notes(cx: Connection, question: dict, plan: dict, result,
                     fields: list[str], games: list[dict]) -> dict:
    """Read availability fields off the injury notes for a pinned game.

    injury_notes already stores player_id, game_id and status as columns, so the
    only thing that has to be read out of the prose is the stated reason. Using
    the column for status rather than re-deriving it from the sentence keeps the
    authoritative value authoritative.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param plan: the validated plan
    :param result: ExecResult to record note evidence into
    :param fields: return-block fields to fill
    :param games: candidate games
    :return: field name -> value
    """
    out: dict = {}
    person_id = plan["row_filters"].get("person_id")
    notes = load_notes(cx, [g["game_id"] for g in games], person_id)
    if not notes:
        return out
    note = notes[0]

    for field in fields:
        if field in note and note[field] is not None:
            out[field] = note[field]
            continue
        reason = regex_reason(note["body"])
        if reason is None:
            parsed = llm_extract(note["body"], [field], question["question"])
            raw = parsed.get(field)
            # Only accept prose the note actually contains.
            if isinstance(raw, str) and raw.strip() and raw.strip().lower() in note["body"].lower():
                reason = raw.strip()
        if reason is not None:
            out[field] = reason

    if out:
        result.notes.add(int(note["note_id"]))
        chosen = [g for g in games if int(g["game_id"]) == int(note["game_id"])]
        if chosen:
            result.games.clear()
            result.touch_game(chosen[0])
    return out


def team_named_in(sentence: str, game: dict, teams: dict) -> int | None:
    """Identify which of a game's two teams a sentence attributes a play to.

    The recaps say who a player belongs to right in the sentence, which is the
    most direct evidence available and better than inferring it from the
    question. Returns None when the sentence names both teams or neither.

    :param sentence: the recap sentence
    :param game: the game row, for its two team ids
    :param teams: team_id -> team dict
    :return: the team_id named, or None if it is not unambiguous
    """
    named = []
    for key in ("home_team_id", "away_team_id"):
        team = teams.get(int(game[key]))
        if not team:
            continue
        labels = [f"{team['city']} {team['name']}", team["name"], team["city"],
                  team["abbreviation"]]
        if any(whole_word_match(lbl, sentence) for lbl in labels):
            named.append(int(game[key]))
    return named[0] if len(named) == 1 else None


def _fill_from_recap(cx: Connection, question: dict, result, fields: list[str],
                     games: list[dict], subject: dict | None,
                     numeric_field: str | None, teams: dict) -> dict:
    """Read narrative fields off a single game's recap.

    A person named in a recap is resolved against that game's box score before it
    is accepted, and the resolved player's row is cited. That is what makes a
    question like "who hit the go-ahead basket" answerable with a
    player_box_scores citation at all: the recap names a surname, and the roster
    turns it into a person.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param result: ExecResult to record recap and box evidence into
    :param fields: return-block fields to fill
    :param games: candidate games; only a single-game set is handled here
    :param subject: the team the question is about
    :param numeric_field: a numeric field, if one was identified
    :return: field name -> verified value
    """
    out: dict = {}
    if len(games) != 1:
        return out
    game = games[0]
    gid = int(game["game_id"])
    recaps = load_recaps(cx, [gid])
    if gid not in recaps:
        return out
    recap = recaps[gid]

    parsed = llm_extract(recap["body"], fields, question["question"])
    roster = game_roster(cx, gid)
    subject_team_id = int(subject["team_id"]) if subject else None

    for field in fields:
        raw = parsed.get(field)
        # The templated sentences are read with a pattern first. The model is
        # only needed for wording the templates miss, and on at least one recap
        # it returned nothing at all for a name that was plainly written there.
        if question["return"].get(field) == "str":
            asks_go_ahead = re.search(GO_AHEAD_TRIGGER, question["question"], re.IGNORECASE)
            hit = regex_person(recap["body"], field, question["question"])
            if hit is not None:
                name, sentence = hit
                # The sentence usually says whose player it was -- "Stroud
                # delivered the go-ahead score for the Saltmere Corsairs" --
                # and that beats the question's subject team for telling apart
                # two players who share a surname across opposing benches.
                attributed = team_named_in(sentence, game, teams) or subject_team_id
                player = resolve_name_in_game(name, roster, sentence, attributed)
                if player is not None:
                    out[field] = player["full_name"]
                    result.box.add((gid, int(player["person_id"])))
                    continue
            # A recap that never mentions a go-ahead play has not got one to
            # report: a 43-point win has no late lead change. Asking the model
            # anyway produced a name, which is the invention this whole pipeline
            # is built to avoid.
            if asks_go_ahead and not re.search(GO_AHEAD_PRESENT, recap["body"], re.IGNORECASE):
                continue
        if raw is None:
            continue
        if question["return"].get(field) == "int":
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            # A number is only trusted if the passage states it.
            if re.search(rf"(?<!\d){value}(?!\d)", recap["body"]):
                out[field] = value
            continue

        if not isinstance(raw, str) or not raw.strip():
            continue
        sentence = next((s for s in split_sentences(recap["body"])
                         if raw.split()[-1].lower() in s.lower()), "")
        player = resolve_name_in_game(raw, roster, sentence, subject_team_id)
        if player is not None:
            out[field] = player["full_name"]
            result.box.add((gid, int(player["person_id"])))
        elif raw.strip().lower() in recap["body"].lower():
            out[field] = raw.strip()

    if out:
        result.recaps.add(int(recap["recap_id"]))
        result.games.clear()
        result.touch_game(game)
    return out


def answer_by_plan(cx: Connection, question: dict, ctx: dict,
                   strict: bool = True) -> dict | None:
    """Answer one question by planning a query and executing it.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param ctx: loaded teams/players lookups and lists, built once per run
    :param strict: when True (the batch path), every declared field must be
        produced and every declared evidence table must be cited, or the
        question abstains. Relaxed for the free-text chat endpoint, whose
        synthesized `return` block lists every field and table a question
        *might* use rather than the ones it does -- there, an unfilled field is
        simply left at its default instead of sinking the answer.
    :return: a result dict, or None if the question could not be planned or the
        data could not support an answer (the caller then abstains)
    """
    return_spec = question["return"]

    # Refuse before planning when the question asks for something these tables
    # do not hold. Planning it anyway produces a query with the impossible part
    # quietly dropped, and a whole-game total answered as a quarter.
    # The question must name something this pipeline has an operation for.
    # Checked before anything else, because a question asking for a player's age
    # or height names no operation at all -- and the demo's fallback shape used
    # to assume any unrecognized question was about scoring, so "how old is
    # Tavian Belcourt" was answered with his point total.
    if not supported_concept(question["question"]):
        return NO_ROWS

    blocked = unsupported_concept(question["question"])
    if blocked is not None:
        return NO_ROWS

    resolved = resolve_question(question["question"], ctx["team_list"],
                                ctx["player_list"], ctx["season"])

    # A name that matched nothing means the filter it should have produced does
    # not exist, and every remaining filter would run without it. That is how a
    # question about a fictional player came back with the league's entire point
    # total, so an unrecognized name ends the question here.
    if unresolved_proper_nouns(question["question"], resolved):
        return NO_ROWS

    # The same check by grammar rather than capitalization, so a lowercase or
    # misspelled name is caught too. Without it, a question about a player who
    # does not exist produced no filter and summed the entire league.
    if unresolved_subject(question["question"], resolved):
        return NO_ROWS

    # The same check again by vocabulary rather than by grammar. The slot above
    # needs a verb to hook onto, so "what is lebron james average 3 pointer"
    # passed it and was answered with a league-wide average across all 841
    # games. Naming a real team alongside the invented player is enough to put
    # that question back inside a plausible scope, so it has to be caught here,
    # before the plan is built.
    if unrecognized_name_phrase(question["question"], resolved):
        return NO_ROWS

    prefill = prefill_plan(question, resolved)
    prompt = build_prompt(question, resolved, prefill)

    plan = None
    for _ in range(PLAN_ATTEMPTS):
        parsed = extract_json(ollama_generate(LLM_MODEL, prompt))
        if parsed is not None:
            plan = validate_plan(parsed, resolved, question)
            if plan is not None:
                break
    # A model that cannot produce a usable plan does not sink the question: the
    # deterministic skeleton is often complete enough on its own, since the
    # entities, dates and thresholds in it were never the model's work.
    if plan is None:
        plan = prefill
        plan["outputs"] = default_outputs(question, plan)
    plan = apply_inferred(plan, question, resolved)

    # A partially-filled `outputs` is common: the model names the fields it
    # understood and drops the rest. Fill the gaps by field name rather than
    # discarding an otherwise good plan. The aggregate is copied first so that
    # deriving defaults cannot rewrite the aggregate the model chose.
    plan = finalize_outputs(plan, question)
    # Outputs are settled now, so the aggregate can be made to match them.
    plan = enforce_output_coherence(plan, question)

    # Everything the question states has to be doing work in the query. A
    # constraint that was recognized and then went nowhere means the plan
    # answers a broader question, and its answer is wrong no matter how exactly
    # it was computed.
    dropped = audit_plan(plan, question, resolved)
    if dropped:
        return NO_ROWS

    # A question that named a player only by surname has not been pinned to a
    # person yet. Resolve it against the games the filters already selected,
    # where the surname is usually unique, rather than aggregating over everyone
    # who happens to share it -- without this, "how many points did Hadderley
    # score in the opener" summed the whole Outriders roster and reported the
    # team's 101 points as one player's.
    # Injury notes are filtered by player too, so a surname left unresolved
    # there is the same failure: "why was Novak injured" found five Novaks,
    # pinned none of them, and reported the status on one of all 2,627 notes in
    # the season as though it were his.
    surname_only = (plan["source"] in ("player_box_scores", "injury_notes")
                    and plan["row_filters"].get("person_id") is None
                    and resolved["players"] and not resolved["players"][0]["exact"])
    if surname_only:
        candidate_games = select_games(cx, plan["game_filters"], plan["game_select"])
        person_id = disambiguate_person(
            cx,
            [p["player_id"] for p in resolved["players"]],
            [int(g["game_id"]) for g in candidate_games],
            plan["row_filters"].get("team_id"),
            plan["row_filters"].get("exclude_team_id"),
            plan["source"],
        )
        if person_id is None:
            return NO_ROWS
        plan["row_filters"]["person_id"] = person_id

    # No qualifying rows is a finding, not a failure. Reported as NO_ROWS so the
    # caller does not then ask a language model the same question and accept a
    # guess: the tables have already said there is nothing to report.
    result = execute_plan(cx, plan, ctx["players"], ctx["teams"])
    if result.empty:
        return NO_ROWS

    declared_shapes = return_spec.get("evidence", [])
    declared_tables = {s["table"] for s in declared_shapes if "table" in s}
    text_fields = [f for f, primitive in plan["outputs"].items() if primitive == "text"]

    # A question that declares a prose table but whose plan named no text field
    # still has to read the prose -- otherwise it would be answered off the
    # tables alone and cite a recap it never opened.
    if declared_tables & TEXT_TABLES and not text_fields:
        text_fields = [f for f in return_spec
                       if f not in ("answerable", "evidence") and f not in plan["outputs"]]

    extracted = {}
    if declared_tables & TEXT_TABLES:
        extracted = fill_text_fields(cx, question, plan, result, ctx["teams"], text_fields)
        # The passage was located and says nothing about what was asked. That is
        # a finding, not a failure to plan -- returning None here sent the
        # question to the similarity fallback, which read the same recap and
        # produced a go-ahead scorer for a game that never had a lead change.
        if text_fields and not any(f in extracted for f in text_fields):
            return NO_ROWS

    zeros = {"bool": False, "int": 0, "float": 0.0, "number": 0.0, "str": ""}
    out = {}
    for field, type_name in return_spec.items():
        if field in ("answerable", "evidence"):
            continue
        if field in extracted:
            value = extracted[field]
        else:
            primitive = plan["outputs"].get(field)
            if primitive is None or primitive == "text" or primitive not in result.values:
                # A declared field the data cannot supply means the answer is
                # incomplete, and an incomplete answer is
                # worth less than an honest abstention.
                if strict:
                    return None
                out[field] = zeros[type_name]
                continue
            value = result.values[primitive]
        try:
            out[field] = coerce_type(value, type_name)
        except (TypeError, ValueError):
            if strict:
                return None
            out[field] = zeros[type_name]

    out = normalize_team_names(out, ctx["team_list"])
    out = normalize_player_names(out, ctx["player_list"])

    # Grounding is never relaxed, whatever `strict` says. `strict` governs how
    # complete an answer has to be; this governs whether it is supported at all.
    # Letting it slide is how "how many points did <player who did not play>
    # score" came back with a number and a citation to a game he was not in.
    evidence = build_evidence(result, declared_shapes)
    if not evidence_fully_grounded(declared_shapes, evidence):
        return None

    out["answerable"] = True
    out["evidence"] = evidence
    return out


def answer_by_similarity(cx: Connection, question: dict, ctx: dict,
                         strict: bool = True) -> dict | None:
    """Fallback: answer from retrieved documents when a question can't be planned.

    Kept because the plan DSL is deliberately small, and a question shaped in a
    way it does not cover should still get an attempt rather than an automatic
    abstention. Retrieval spans all four embedded tables, and the model's cited
    evidence is checked against what was actually retrieved -- nothing it names
    that was not in the context can be cited.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param ctx: loaded teams/players lookups and lists
    :return: a result dict, or None to abstain
    """
    return_spec = question["return"]
    declared_shapes = return_spec.get("evidence", [])
    declared = {s["table"] for s in declared_shapes if "table" in s}
    qvec = ollama_embed(EMBED_MODEL, QUERY_PREFIX + question["question"])

    id_columns = {"game_details": "game_id", "player_box_scores": "game_id",
                  "game_recaps": "recap_id", "injury_notes": "note_id"}
    blocks, retrieved = [], {}
    for table in declared:
        rows = retrieve(cx, qvec, TOP_K, table, id_columns.get(table, "game_id"))
        if rows:
            blocks.append(build_context(rows, table))
            retrieved[table] = {int(r["id"]) for r in rows}
    if not blocks:
        return None

    prompt = (
        "You answer questions about basketball games using only the context below. "
        "Respond with a single JSON object matching this shape (types shown, not values): "
        f"{json.dumps(return_spec)}\n"
        "Every evidence row you cite must be one you can see in the context below. "
        "If the context doesn't contain enough information to answer, set \"answerable\": false "
        "and return your best-effort defaults for the other fields with an empty evidence list. "
        "Use team mascots only, e.g. \"Outriders\" instead of \"Sablecrest Outriders (SBC)\". "
        "Give player names exactly as \"First Last\". "
        "Format any score as \"<winner points>-<loser points>\", e.g. \"117-102\".\n\n"
        f"Context:\n" + "\n".join(blocks) + f"\n\nQuestion: {question['question']}\nJSON:"
    )
    parsed = extract_json(ollama_generate(LLM_MODEL, prompt))
    if parsed is None or not parsed.get("answerable", False):
        return None

    zeros = {"bool": False, "int": 0, "float": 0.0, "number": 0.0, "str": ""}
    out = {}
    for field, type_name in return_spec.items():
        if field in ("answerable", "evidence"):
            continue
        try:
            out[field] = coerce_type(parsed[field], type_name)
        except (KeyError, TypeError, ValueError):
            if strict:
                return None
            out[field] = zeros[type_name]

    out = normalize_team_names(out, ctx["team_list"])
    out = normalize_player_names(out, ctx["player_list"])

    # Keep only citations that name a row actually present in the context.
    evidence = []
    for entry in parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else []:
        if not isinstance(entry, dict):
            continue
        shape = next((s for s in declared_shapes if s.get("table") == entry.get("table")), None)
        if shape is None:
            continue
        try:
            item = {"table": entry["table"]}
            for key, type_name in shape.items():
                if key != "table":
                    item[key] = coerce_type(entry[key], type_name)
        except (KeyError, TypeError, ValueError):
            continue
        identifier = item.get("id", item.get("game_id"))
        if identifier in retrieved.get(item["table"], set()):
            evidence.append(item)

    if not evidence_fully_grounded(declared_shapes, evidence):
        return None
    out["answerable"] = True
    out["evidence"] = evidence
    return out


# This file has to work on any questions.json with different wording, entities
# and ids, not just the examples in questions/. Do not branch on
# question id or match question text with patterns; every question must go
# through the same retrieve/answer path, driven only by its `return` block.
def answer_question(cx: Connection, question: dict, ctx: dict,
                    strict: bool = True) -> dict:
    """Answer one question, abstaining rather than guessing.

    Every failure route ends in abstain_result(): an entity that resolves to
    nothing, a plan that will not validate, a query that matches no rows, a
    prose fact that cannot be verified, or evidence that does not cover every
    table the question declared. The README scores a confident number for a
    missing row at zero, so an abstention is never the worse trade.

    :param cx: open SQLAlchemy connection
    :param question: one entry from questions/questions.json
    :param ctx: loaded teams/players lookups and lists
    :param strict: see answer_by_plan; False only for the free-text endpoint
    :return: a result dict matching question["return"]'s shape
    """
    result = answer_by_plan(cx, question, ctx, strict)
    if result is NO_ROWS:
        return abstain_result(question["return"])
    if result is None:
        result = answer_by_similarity(cx, question, ctx, strict)
    return result if result is not None else abstain_result(question["return"])


def load_context(cx: Connection) -> dict:
    """Load the lookups every question needs, once per run.

    :param cx: open SQLAlchemy connection
    :return: dict with teams/players as both id-keyed lookups and lists
    """
    team_list = load_teams(cx)
    player_list = load_players(cx)
    return {
        "team_list": team_list,
        "player_list": player_list,
        "teams": {t["team_id"]: t for t in team_list},
        "players": {p["player_id"]: p for p in player_list},
        "season": load_season_bounds(cx),
    }


def main() -> None:
    """Answer every question in questions/questions.json and write questions/answers.json."""
    print("Starting RAG Answering")
    eng = sa.create_engine(DB_DSN)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    outputs = []
    with eng.begin() as cx:
        ctx = load_context(cx)
        for q in questions:
            result = answer_question(cx, q, ctx)
            outputs.append({"id": q["id"], "result": result})
            print(f"Q{q['id']}: {q['question']}\n  -> {json.dumps(result)}")

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Finished RAG Answering: wrote {len(outputs)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
