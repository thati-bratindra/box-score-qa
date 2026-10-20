"""Pull facts out of recap and injury-note prose, then verify them.

Some facts exist nowhere but the text: why a player sat, who hit the go-ahead
shot, how big a deficit a team erased. This module is the only place the LLM is
allowed to read free text, and every value it produces is checked against the
tables before it can reach an answer.

The checking matters because of how the corpus is written. Recaps name players by
last name ("Ivanov came up with the go-ahead basket"), and 818 of 1682 game-team
rosters contain two players who share a last name -- so a name lifted from a
recap is not an identity. resolve_name_in_game turns it into one, or refuses.

Numbers take a regex fast path. The recaps are generated from templates (four
phrasings cover every comeback in the data), so a pattern match is both instant
and perfectly reproducible, which matters for questions that have to scan every
game in a season. The LLM is the fallback for text no pattern fits, not the
default.

README rule enforced here: the tables are authoritative and a recap can be
wrong. Where the two disagree, the table wins; where they cannot be reconciled,
the caller abstains.
"""
import json
import re

from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.config import LLM_MODEL
from backend.resolve import whole_word_match
from backend.utils import ollama_generate

# Numeric narrative facts the recap corpus actually contains. Derived from the
# text itself (every comeback in the data uses one of four templates), not from
# any particular question, so a differently-worded question about the same fact
# still hits them.
# Each verb gets a slot for the words the corpus puts between it and the
# number -- "an early 12-point deficit", "by just 1 point", "by as many as 14".
# Writing them out one phrasing at a time is what let these drift: the four
# original patterns covered 436 sentences and missed 110 quantified ones,
# including the 14-point hole that made a deficit answer read 13.
# backend/validate.py now scans the corpus against these and fails on a miss.
DEFICIT_PATTERNS = [
    # One pattern, not one per phrasing. The verb is matched by stem so every
    # inflection counts -- "erased", "erasing", "dug out of", "dig out of",
    # "overcame", "overcoming" all describe the same thing, and listing them
    # individually is what kept leaving holes.
    r"\b(?:rall|battl|fought|fight|climb|dug|dig|overcam|overcom|eras|fac|came|com"
    r"|shak|shook|surviv|withstood|weather|claw|storm|charg|surg|bounc|roar)\w*"
    r"(?:\s+(?:back|off|out|of|from)){0,3}\s+an?\s+(?:\w+\s+){0,2}?"
    r"(\d+)[-\s]point\s+(?:deficit|hole)",
    # Said without the word "deficit" at all, which is how 160 recaps write it.
    r"(?:trailed|trailing|down)\s+by\s+(?:\w+\s+){0,3}?(\d+)\s+points?",
]
LEAD_PATTERNS = [
    r"led by as many as\s+(\d+)\s+points",
    r"ahead by\s+(\d+)\s+points",
    r"advantage to as much as\s+(\d+)\s+points",
    r"stretched its lead to\s+(\d+)",
    # "Built a lead as big as 21 points" is the same fact in the corpus's other
    # voice, and appears in 65 recaps that reported no lead at all. The second
    # pattern covers the nouns the corpus uses for a lead -- cushion, edge,
    # advantage -- and the verbs it uses for growing one.
    r"(?:lead|advantage|cushion|edge)[^.]{0,24}?as\s+(?:many|much|big|large)\s+as\s+(\d+)",
    r"(?:built|open|push|extend|stretch|swell|grew|grow)\w*\s+"
    r"(?:\w+\s+){0,2}?(?:lead|advantage|cushion|edge)\s+to\s+(\d+)",
]

# Which pattern set applies is decided by the vocabulary of the question itself,
# not by which question it is. "Deficit", "comeback" and "rallied" all describe
# the same quantity in this corpus.
CONCEPT_PATTERNS = {
    "deficit": (DEFICIT_PATTERNS,
                r"deficit|comeback|came back|rallied|overcame|erased|hole"
                r"|trailed|trailing|down by"),
    "lead": (LEAD_PATTERNS, r"\blead\b|\bled\b|ahead|advantage|margin at one"),
}

# The recaps name the go-ahead scorer with a handful of templates. Reading them
# with a pattern is both faster and steadier than asking the model, which on one
# recap returned nothing usable and turned an answerable question into an
# abstention. The model stays as the fallback for wording these do not cover.
GO_AHEAD_PATTERNS = [
    r"go-ahead\s+(?:bucket|basket|score|points?|shot)[^.]*?"
    r"\b(?:came|got it|got the)?\s*from\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)",
    # "…and got the late go-ahead bucket from Amari Trask, who scored…"
    r"(?:\w+\s+){0,2}?go-ahead\s+(?:bucket|basket|score|shot)\s+from\s+"
    r"([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)",
    # Participle voice: "…with Varga supplying the go-ahead basket."
    r"\b([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\s+(?:supplying|delivering|providing|hitting|"
    r"making|scoring)\s+the\s+(?:\w+\s+){0,2}?(?:go-ahead|winning|decisive)",
    # The verb may be separated from "the go-ahead" by a whole other object:
    # "Trask delivered a 30-point performance and the late go-ahead bucket".
    r"\b([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\s+(?:delivered|supplied|provided|hit|made|"
    r"scored|knocked down|came up with)\s+(?:both\s+)?(?:[^.]{0,44}?\band\s+)?"
    r"the\s+(?:\w+\s+){0,2}?go-ahead",
    r"\b([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\s+(?:delivered|supplied|provided|hit|made|"
    r"scored)\s+the\s+(?:\w+\s+){0,2}?(?:winning|decisive)\s+(?:basket|bucket|score|shot)",
    # The possessive voice: "Ferreira's late go-ahead bucket sent the Outriders
    # past...". The verb patterns above cannot see it, and it is how 25 recaps
    # name the scorer -- the only place some of them name anybody.
    r"\b([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)['’]s\s+(?:\w+\s+){0,2}?"
    r"(?:go-ahead|winning|decisive)\s+(?:basket|bucket|score|shot|jumper|points?)",
    # "It was Varga who put the Outriders ahead for good late in the fourth."
    r"\b([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\s+(?:who\s+)?put\s+the\s+[^.]*?ahead for good",
]
GO_AHEAD_TRIGGER = r"go-ahead|winning (?:basket|bucket|shot|score)|decisive"
# Used to decide a recap has *no* go-ahead play to report. Narrower than the
# trigger above on purpose: recaps call plenty of ordinary wins "decisive", and
# treating that as a late lead change let a 43-point rout produce a go-ahead
# scorer who did not exist.
GO_AHEAD_PRESENT = r"go-ahead|winning (?:basket|bucket|shot|score)|game[- ]winner"


def regex_person(body: str, field: str, question: str) -> tuple[str, str] | None:
    """Pull a person's name out of a templated recap sentence.

    Only handles the go-ahead/game-winner templates, which is where the corpus
    actually names an individual for a play rather than a stat line.

    :param body: recap text
    :param field: the return-block field being filled
    :param question: the original question, to check it is asking about this
    :return: (name, the sentence it came from), or None
    """
    if not re.search(GO_AHEAD_TRIGGER, f"{question} {field}", re.IGNORECASE):
        return None
    for sentence in split_sentences(body):
        if not re.search(GO_AHEAD_TRIGGER, sentence, re.IGNORECASE):
            continue
        for pattern in GO_AHEAD_PATTERNS:
            m = re.search(pattern, sentence)
            if m:
                return m.group(1), sentence
    return None


# A stat a recap sentence asserts, used to tell two players with the same last
# name apart by checking which one actually put up that line.
STAT_HINT_PATTERNS = [
    (r"(\d+)\s+points?", "points"),
    (r"(\d+)\s+rebounds?", "rebounds"),
    (r"(\d+)\s+assists?", "assists"),
]

MAX_LLM_EXTRACTIONS = 60


def split_sentences(body: str) -> list[str]:
    """Split recap text into sentences.

    :param body: recap or note text
    :return: sentences, whitespace-stripped
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body or "") if s.strip()]


def team_labels(team: dict) -> list[str]:
    """Every surface form a team appears under in recap prose.

    Recaps mix full names ("the Sablecrest Outriders"), cities ("for Sablecrest") and
    abbreviations ("the ANC and the TUL met"), so all of them have to match.

    :param team: a team dict from resolve.load_teams()
    :return: label strings to test with whole_word_match
    """
    return [f"{team['city']} {team['name']}", team["name"], team["city"], team["abbreviation"]]


def pick_concept(question_text: str, field: str) -> str | None:
    """Decide which numeric pattern set a question is asking about.

    Matches on the question's own vocabulary and on the name of the field the
    question's `return` block declares -- both of which vary freely from one question
    to the next -- rather than on question identity.

    :param question_text: the question being answered
    :param field: the return-block field the value will fill
    :return: a key of CONCEPT_PATTERNS, or None if nothing matches
    """
    haystack = f"{question_text} {field}".lower()
    for concept, (_, trigger) in CONCEPT_PATTERNS.items():
        if re.search(trigger, haystack):
            return concept
    return None


def regex_number(body: str, concept: str, subject: dict | None) -> int | None:
    """Extract a narrative number from one recap, if a template matches.

    When a subject team is given, only sentences naming that team count. A recap
    describes both sides, and "the Foxhounds rallied from 12 down" says nothing
    about what the Outriders overcame -- attributing it to the wrong team is the
    main way this kind of extraction goes wrong.

    :param body: recap text
    :param concept: key of CONCEPT_PATTERNS
    :param subject: the team the fact must be about, or None for any
    :return: the largest matching number, or None
    """
    patterns, _ = CONCEPT_PATTERNS[concept]
    labels = team_labels(subject) if subject else None
    found = []
    for sentence in split_sentences(body):
        if labels and not any(whole_word_match(lbl, sentence) for lbl in labels):
            continue
        for pattern in patterns:
            found += [int(m) for m in re.findall(pattern, sentence, re.IGNORECASE)]
    return max(found) if found else None


def load_recaps(cx: Connection, game_ids: list[int]) -> dict[int, dict]:
    """Load recaps for a set of games.

    :param cx: open SQLAlchemy connection
    :param game_ids: games to load recaps for
    :return: game_id -> {recap_id, body}
    """
    if not game_ids:
        return {}
    sql = ('SELECT recap_id, game_id, "text" AS body FROM game_recaps '
           "WHERE game_id = ANY(:gids) ORDER BY game_id, recap_id")
    rows = cx.execute(text(sql), {"gids": [int(g) for g in game_ids]}).mappings().all()
    return {int(r["game_id"]): {"recap_id": int(r["recap_id"]), "body": r["body"]} for r in rows}


def load_notes(cx: Connection, game_ids: list[int], person_id: int | None) -> list[dict]:
    """Load injury notes for a set of games, optionally for one player.

    injury_notes already carries player_id, game_id, team_id and status as
    columns, so identity and availability are a lookup -- only the reason for the
    absence has to be read out of the prose.

    :param cx: open SQLAlchemy connection
    :param game_ids: games to load notes for
    :param person_id: restrict to one player, or None for all
    :return: note rows ordered by (game_id, note_id)
    """
    if not game_ids:
        return []
    where = ["n.game_id = ANY(:gids)"]
    params: dict = {"gids": [int(g) for g in game_ids]}
    if person_id is not None:
        where.append("n.player_id = :pid")
        params["pid"] = int(person_id)
    sql = ('SELECT n.note_id, n.game_id, n.player_id, n.team_id, n.note_date, '
           'n.status, n."text" AS body FROM injury_notes n WHERE '
           + " AND ".join(where) + " ORDER BY n.game_id, n.note_id")
    return [dict(r) for r in cx.execute(text(sql), params).mappings().all()]


def regex_reason(body: str) -> str | None:
    """Pull the stated reason out of an injury note.

    The notes use a handful of templates -- a parenthetical "(ankle)",
    "because of X", "due to X", "cites X", and "with X" -- which between them
    cover the corpus, so the reason is recoverable without a model call.

    :param body: note text
    :return: the reason phrase, or None if no template matched
    """
    for pattern in (r"\(([^)]{2,40})\)",
                    r"because of\s+([^.,;]{2,60})",
                    r"\bdue to\s+([^.,;]{2,60})",
                    r"cites\s+([^.,;]{2,60})",
                    r"\btag for\s+([^.,;]{2,60})",
                    r"\bwith\s+([^.,;]{2,60})"):
        m = re.search(pattern, body or "", re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def game_roster(cx: Connection, game_id: int) -> list[dict]:
    """Load everyone who appeared in a game, with names and their stat line.

    :param cx: open SQLAlchemy connection
    :param game_id: the game
    :return: rows with person_id, team_id, points, rebounds, assists, names
    """
    sql = (
        "SELECT b.person_id, b.team_id, b.points, b.assists, "
        "b.offensive_reb + b.defensive_reb AS rebounds, "
        "p.first_name, p.last_name, p.first_name || ' ' || p.last_name AS full_name "
        "FROM player_box_scores b JOIN players p ON p.player_id = b.person_id "
        "WHERE b.game_id = :gid ORDER BY b.person_id"
    )
    return [dict(r) for r in cx.execute(text(sql), {"gid": int(game_id)}).mappings().all()]


def resolve_name_in_game(name: str, roster: list[dict], sentence: str = "",
                         team_id: int | None = None) -> dict | None:
    """Turn a name mentioned in prose into a specific player, or give up.

    Tried in order, each step narrower than the last:

    1. Exact full name against the roster -- unambiguous when the recap gives one.
    2. Last name plus the team the recap attributes the play to. Accepted only if
       it leaves exactly one player; roughly half of all game-team rosters
       contain a shared last name, so "unique" has to be checked, not assumed.
    3. Last name plus a stat asserted in the same sentence ("scored 35"), matched
       against the box score. This is the tie-breaker that also doubles as a
       consistency check between the recap and the tables.
    4. Give up. An unresolvable name makes the caller abstain, which is correct:
       there is no way to cite a player_box_scores row for a player we cannot
       identify.

    :param name: the name as written in the text
    :param roster: rows from game_roster()
    :param sentence: the sentence the name appeared in, for the stat hint
    :param team_id: the team the text attributes the play to, if known
    :return: the matching roster row, or None
    """
    cleaned = re.sub(r"\s*\([^)]*\)", "", (name or "")).strip()
    if not cleaned:
        return None

    exact = [r for r in roster if r["full_name"].lower() == cleaned.lower()]
    if len(exact) == 1:
        return exact[0]

    surname = cleaned.split()[-1]
    candidates = [r for r in roster if r["last_name"].lower() == surname.lower()]
    if not candidates:
        candidates = [r for r in roster if whole_word_match(r["last_name"], cleaned)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    if team_id is not None:
        narrowed = [r for r in candidates if int(r["team_id"]) == int(team_id)]
        if len(narrowed) == 1:
            return narrowed[0]
        if narrowed:
            candidates = narrowed

    for pattern, column in STAT_HINT_PATTERNS:
        for raw in re.findall(pattern, sentence or "", re.IGNORECASE):
            matched = [r for r in candidates if int(r[column] or 0) == int(raw)]
            if len(matched) == 1:
                return matched[0]
    return None


def llm_extract(body: str, fields: list[str], question: str) -> dict:
    """Ask the model for specific fields out of one passage.

    Kept deliberately narrow: one passage, a named list of fields, JSON out. The
    model is never asked to decide whether the passage is relevant or to combine
    passages -- the structured filters already did that.

    :param body: the passage to read
    :param fields: field names to fill
    :param question: the original question, for context
    :return: parsed dict, empty if the response was unusable
    """
    prompt = (
        "Extract facts from the passage. Respond with one JSON object containing "
        f"exactly these keys: {json.dumps(fields)}.\n"
        "Use only what the passage states. Use null for anything it does not say. "
        "For a person, give the name exactly as the passage writes it.\n\n"
        f"Passage:\n{body}\n\nQuestion: {question}\n\nJSON:"
    )
    raw = ollama_generate(LLM_MODEL, prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def scan_recaps_for_number(cx: Connection, games: list[dict], concept: str,
                           subject: dict | None, question: str,
                           field: str) -> dict[int, tuple[int, int]]:
    """Read a narrative number out of every candidate game's recap.

    This is the shape no amount of similarity search can answer: "the largest
    deficit overcome all season" is a maximum over the whole candidate set, so
    every recap in it has to be read. Top-k retrieval would simply return the
    five recaps that read most like the question.

    Regex handles the templated majority instantly and identically on every run.
    The model is called only for recaps no pattern fits, and only up to a cap, so
    a season-wide scan stays affordable.

    :param cx: open SQLAlchemy connection
    :param games: candidate game rows from execute.select_games()
    :param concept: key of CONCEPT_PATTERNS
    :param subject: the team the fact must be about
    :param question: the original question, passed to the model fallback
    :param field: the return-block field being filled
    :return: game_id -> (value, recap_id)
    """
    recaps = load_recaps(cx, [g["game_id"] for g in games])
    found: dict[int, tuple[int, int]] = {}
    misses: list[int] = []

    for game in games:
        gid = int(game["game_id"])
        recap = recaps.get(gid)
        if not recap:
            continue
        value = regex_number(recap["body"], concept, subject)
        if value is not None:
            found[gid] = (value, recap["recap_id"])
        else:
            misses.append(gid)

    # The model is a fallback for phrasing the patterns do not cover, not for
    # recaps that simply have nothing to say. Most misses are the latter: a game
    # with no comeback has no comeback sentence. Asking the model anyway is how
    # a season's worth of ordinary wins turns into a season's worth of invented
    # deficits -- an early version of this did exactly that, reporting comebacks
    # in six games the team had lost.
    #
    # So the fallback only runs when no pattern matched anywhere (suggesting the
    # corpus is worded differently than expected), and anything it returns must
    # be corroborated in the text: the number has to appear in a sentence that
    # names the subject team and uses the vocabulary of the concept. A value the
    # passage does not actually state is discarded.
    if not found:
        _, trigger = CONCEPT_PATTERNS[concept]
        labels = team_labels(subject) if subject else None
        for gid in misses[:MAX_LLM_EXTRACTIONS]:
            recap = recaps[gid]
            parsed = llm_extract(recap["body"], [field], question)
            try:
                value = int(parsed.get(field))
            except (TypeError, ValueError):
                continue
            for sentence in split_sentences(recap["body"]):
                if labels and not any(whole_word_match(lbl, sentence) for lbl in labels):
                    continue
                if not re.search(trigger, sentence, re.IGNORECASE):
                    continue
                if re.search(rf"(?<!\d){value}(?!\d)", sentence):
                    found[gid] = (value, recap["recap_id"])
                    break
    return found
