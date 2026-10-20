"""FastAPI chat endpoint over the same answer path used by backend/rag.py.

The endpoint runs the full pipeline -- resolve, plan, execute, extract, verify --
rather than a separate prompt, so what the demo shows is what the batch run
does, including its abstentions. A free-text question has no `return` block, so
one is synthesized to ask for a general answer, and the same machinery fills it.

usage:
    uvicorn backend.server:app --reload
"""
import re

import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import DB_DSN
from backend.plan import PERSON_QUESTION, infer_field
from backend.rag import answer_question, load_context

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)

GAME = {"table": "game_details", "id": "int"}
BOX = {"table": "player_box_scores", "game_id": "int", "person_id": "int"}
RECAP = {"table": "game_recaps", "id": "int"}
NOTE = {"table": "injury_notes", "id": "int"}

# A free-text question arrives with no `return` block, and the pipeline is driven
# by that block: it decides which table to read and which fields to fill. So the
# endpoint has to synthesize the block a written question would have declared.
#
# One catch-all block listing every field and table does not work -- it was the
# first thing tried. Declaring all four evidence tables makes the planner read
# recaps for an injury question, and declaring every field invites mappings for
# fields the question never asked about, so "points in January" came back with a
# deficit and a season total attached. The shape has to be narrowed to the
# question, the way a written question narrows it.
#
# This lives here rather than in rag.py on purpose: it is a shim for the demo's
# free-text box, not part of the batch path, which always has a real block.
# A question asking about a person, however the person is described. Matching a
# fixed list of nouns was not enough: "which Outriders starter" was recognized and
# "which Outriders bench player" was not, so the same question about the other half
# of the roster came back as a bare point total with nobody's name on it. The
# pattern now allows words between "which" and the noun, and the noun list
# covers roles and positions rather than just the word "player".
#
# The bare word "who" is deliberately absent. "…and who won?" and "who beat
# them" ask about a *team*, and matching them here sent two of the eleven public
# questions to a player shape: Q5 answered "Florian Ferreira, 30 points" instead of
# the final score, and Q11 named a player instead of the team that beat them.
# PERSON_QUESTION (shared with the plan audit) scopes "who" to verbs a person
# does, so those two stay out.
PERSON_SUBJECT = (
    PERSON_QUESTION
    + r"|\bwhich\b(?:\s+[\w'-]+){0,4}?\s+"
      r"(?:shooter|shooters|sub|subs|substitute|substitutes|teammate|teammates)\b"
      r"|\bleading scorer\b|\btop scorer\b|\bbench\b"
)
TEAM_SUBJECT = (r"\bwhich team\b|\bwhat team\b|\bwhich opponent\b|\bwhich club\b"
                r"|\bwho won\b|\bwho beat\b|\bwho defeated\b|\bwho lost\b")

# "The most games with 30+ points" counts games, not points, even though the
# question mentions points. Without this Q3 named the right player and then
# reported his point total instead of his game count.
GAME_COUNT_SUBJECT = (r"\b(?:most|fewest)\s+games\b|\bhow many games\b|\bgames with\b"
                      # "the most times" counts occasions, not points. Without
                      # it, "who dropped 30+ on the Gannets the most times"
                      # named the right player and reported his point total.
                      r"|\b(?:most|fewest)\s+times\b|\bhow many times\b"
                      r"|\bmost often\b|\bmost frequently\b")

# "Out" only counts in its availability sense. A bare \bout\b matched "who came
# out on top" -- a question about the final score -- and routed it to the injury
# notes, which answered with the name of the most-injured player in the league.
INJURY_WORDS = (r"\bwhy\b|injur|\bmiss(?:ed|ing|es)?\b|\bavailable\b"
                # "Out of" alone is not about availability: "the biggest hole
                # they climbed out of" is a comeback question, and it was being
                # answered with somebody's sprained ankle.
                r"|\bout\s+(?:for|with)\b|\bout of the (?:game|lineup|rotation)\b"
                r"|\bruled out\b|\b(?:is|was|were|are)\s+out\b"
                r"|\bs(?:at|it)\s+out\b|questionable|doubtful|\bsidelined\b"
                r"|\bunavailable\b|\bdue to rest\b|\bresting\b|\binactive\b"
                r"|\blisted\b")

CHAT_SHAPES = [
    # "Who missed the most games" is an injury question that wants a name, not a
    # status. Checked before the plain injury shape, which answers with the
    # status and reason of one note and never says whose.
    (rf"(?=.*(?:{INJURY_WORDS}))(?=.*(?:\bwho\b|\bwhich\b))",
     {"answerable": "bool", "player_name": "str", "games": "int",
      "evidence": [NOTE, GAME]}),
    (INJURY_WORDS,
     {"answerable": "bool", "status": "str", "reason": "str",
      "evidence": [NOTE, GAME]}),
    # The largest lead is the mirror of the largest deficit and lives in the
    # same recap prose. Checked first, because "the biggest lead they built"
    # contains none of the comeback words below and was falling through to a
    # points shape that had nothing to do with it.
    (r"(?:biggest|largest|greatest|maximum|max)\s+(?:\w+\s+){0,2}?"
     r"(?:lead|advantage|cushion|edge)\b|lead as (?:big|large|much) as"
     r"|(?:built|opened|stretched|extended)\s+(?:a|its|their)\s+(?:biggest\s+)?lead",
     {"answerable": "bool", "lead": "int", "opponent": "str", "score": "str",
      "evidence": [RECAP, GAME]}),
    # "The biggest hole they climbed out of" is a comeback question. Without
    # these words it matched only "final score" further down and was answered
    # with the game's margin instead of the deficit erased.
    (r"go-ahead|deficit|comeback|came back|rallied|overcame|erased|game-winner|clutch"
     r"|\bhole\b|climbed out of|dug out of|fought back|trailed by",
     {"answerable": "bool", "player_name": "str", "deficit": "int",
      "opponent": "str", "score": "str", "evidence": [RECAP, GAME]}),
    # A streak is counted over games and touches no box score rows, so it needs
    # a game-only shape. Under the default shape it computed the right answer
    # and then failed grounding, because that shape promises box score evidence
    # a streak never produces.
    (r"\bstreaks?\b|\bconsecutive\b|\bin a row\b|\bstraight\b",
     {"answerable": "bool", "games": "int", "evidence": [GAME]}),
    # Checked before the person shape, since "which team" also starts with
    # "which" and is asking about a club, not a player.
    (TEAM_SUBJECT,
     {"answerable": "bool", "winner": "str", "opponent": "str", "games": "int",
      "margin": "int", "score": "str", "evidence": [GAME]}),
    (PERSON_SUBJECT,
     {"answerable": "bool", "player_name": "str", "points": "int",
      "evidence": [GAME, BOX]}),
    # Asking for the result of a game, however it is phrased. "What was the
    # final and who came out on top" matched none of the earlier shapes once
    # the injury pattern stopped claiming the word "out", and fell through to a
    # player shape that had nothing to say.
    (r"final score|\bthe final\b|\bwho won\b|\bcame out on top\b|\bwho prevailed\b"
     r"|\bwho took (?:it|the game)\b|\bmargin\b|beat them|largest win|largest loss"
     r"|\bwhat was the score\b|\bhow did it end\b|\bresult of\b"
     # A "closest game" or "biggest blowout" is a question about one game's
     # result, not about anybody's point total.
     r"|\bclosest\b|\bnarrowest\b|\btightest\b|\bblowout\b|\bmost lopsided\b",
     {"answerable": "bool", "winner": "str", "opponent": "str", "score": "str",
      "margin": "int", "evidence": [GAME]}),
]
DEFAULT_SHAPE = {"answerable": "bool", "points": "int", "games": "int",
                 "evidence": [GAME, BOX]}


def chat_return_spec(question_text: str) -> dict:
    """Pick the `return` shape a free-text question would have declared.

    The statistic is named after what the question asks about, so a rebounding
    question returns a "rebounds" field rather than reporting a rebound total
    under the label "points" -- the number was right, but the answer read as
    though it were something else.

    :param question_text: the user's question
    :return: a return-block dict for the pipeline to drive from
    """
    # "How many <thing>" names the field outright. Without this, "how many teams
    # did X play for" fell through to the default points-and-games shape and
    # answered with his game count.
    #
    # Adjectives are allowed in front of the noun. Requiring it to sit directly
    # after "how many" meant "how many home games did the Sunhawks win" matched
    # nothing, took the default shape, and led with a point total instead of the
    # count it was asked for. The bound stays at two words so that the noun in
    # "how many points did he score in close games" is still "points".
    counted = re.search(r"\bhow many\s+(?:[\w'-]+\s+){0,2}?"
                        r"(teams|players|games|wins|losses)\b",
                        question_text, re.IGNORECASE)
    if counted:
        field = counted.group(1).lower()
        # "How many games was X injured for" counts absences, not games played,
        # so it has to be answered from the notes. Routing it to game_details
        # left the player with nothing to constrain and counted every game in
        # the season.
        if field == "games" and re.search(INJURY_WORDS, question_text, re.IGNORECASE):
            return {"answerable": "bool", "games": "int", "evidence": [NOTE, GAME]}
        # A game count is read from the schedule, unless the question counts
        # games meeting a condition on one player -- "how many games did he
        # score 30+ points" is settled per box score, and promising schedule
        # evidence alone left the threshold nowhere to live, so the audit
        # abstained on a question the tables answer.
        team_level = (field in ("games", "wins", "losses")
                      and infer_field(question_text, "player_box_scores") is None)
        evidence = [GAME] if team_level else [GAME, BOX]
        return {"answerable": "bool", field: "int", "evidence": evidence}

    # What the answer is counted in: a game count when the question ranks games,
    # otherwise the statistic the question names.
    if re.search(GAME_COUNT_SUBJECT, question_text, re.IGNORECASE):
        stat = "games"
    else:
        stat = infer_field(question_text, "player_box_scores") or "points"

    for pattern, shape in CHAT_SHAPES:
        if re.search(pattern, question_text, re.IGNORECASE):
            shape = dict(shape)
            if "points" in shape and stat != "points":
                shape[stat] = shape.pop("points")
            return shape
    shape = dict(DEFAULT_SHAPE)
    if stat == "games":
        # The question counts occasions, so a point total is not the answer to
        # it. The default shape carries one anyway, and "how many times did he
        # score 30 or more" led with the season total and put the count it
        # asked for behind it.
        shape.pop("points", None)
    elif stat != "points":
        shape[stat] = shape.pop("points")
    return shape


class Q(BaseModel):
    question: str


def describe(result: dict) -> str:
    """Render a result dict as a sentence for the chat UI.

    :param result: a result dict as returned by answer_question()
    :return: a human-readable answer, or an explicit abstention
    """
    if not result.get("answerable"):
        return ("I can't answer that from the data. Either the question names "
                "something the tables don't contain, or no rows support it.")
    # Fields whose names would read oddly on their own get a phrasing; anything
    # else is rendered as "<value> <field name>", so a new stat needs no change
    # here.
    labels = {"deficit": "point deficit", "margin": "point margin",
              "seconds": "seconds played", "wins": "wins", "losses": "losses"}
    skip = {"answerable", "evidence", "player_name", "team", "team_name",
            "status", "reason", "winner", "score", "opponent", "games"}

    parts = []
    for key in ("player_name", "team", "team_name"):
        if result.get(key):
            parts.append(str(result[key]))
            break
    if result.get("status"):
        parts.append(f"status: {result['status']}")
    if result.get("reason"):
        parts.append(f"reason: {result['reason']}")
    for key, value in result.items():
        if key in skip or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if value:
            word = labels.get(key, key)
            # "1 points" reads like a bug even when the number is right.
            if value == 1 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            parts.append(f"{value} {word}")
    # A game count of one says nothing a single-game answer has not said already.
    # But when nothing else has been reported, the count is the answer rather
    # than a qualifier on one: a streak of 20 read as "across 20 games", which
    # describes a span instead of the run of wins that was asked about.
    games = result.get("games")
    if isinstance(games, int) and not isinstance(games, bool):
        if not parts:
            parts.append(f"{games} game" + ("s" if games != 1 else ""))
        elif games > 1:
            parts.append(f"across {games} games")
    if result.get("opponent"):
        parts.append(f"opponent: {result['opponent']}")
    if result.get("winner"):
        parts.append(f"winner: {result['winner']}")
    if result.get("score"):
        parts.append(f"score: {result['score']}")
    return ", ".join(parts) if parts else "Answered, but with no fields to report."


@app.post("/api/chat")
def answer(q: Q) -> dict:
    """Answer a free-text question using the same path as backend/rag.py.

    :param q: request body with a single "question" field
    :return: {"answer": str, "evidence": [...], "result": {...}}
    """
    print(f"Received question: {q.question}")
    question = {"id": 0, "question": q.question, "return": chat_return_spec(q.question)}
    with eng.begin() as cx:
        # strict=False: the synthesized shape is a guess at what was asked, so a
        # field it named but the data cannot fill should be left blank rather
        # than abstaining on an answer the rest of the shape supports.
        result = answer_question(cx, question, load_context(cx), strict=False)
    return {
        "answer": describe(result),
        # Cite whatever the answer actually touched, rather than assuming the
        # evidence is always a game_details row.
        "evidence": result.get("evidence", []),
        "result": result,
    }
