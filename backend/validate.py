"""Validate questions/answers.json against questions/answers.schema.json.

Also checks that the set of ids in answers.json exactly matches the set of
ids in questions/questions.json, since the schema alone won't catch a missing or
duplicated question.

And checks the extraction patterns against the recap corpus. That check exists
because the patterns silently rotted: four deficit templates covered the
phrasings someone thought of and missed 110 sentences that state a deficit in
other words, which made "the largest deficit the Outriders overcame" answer 13
when the real answer, written "Down by 14 points at one stretch", was 14. A
wrong number from a missing pattern looks exactly like a right one, so the only
way to catch it is to ask the corpus rather than to think harder.

usage:
    python -m backend.validate
"""
import json
import os
import re
import sys

import jsonschema
import sqlalchemy as sa

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "questions", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "questions", "answers.json"))
SCHEMA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "questions", "answers.schema.json"))

MAX_ERRORS_SHOWN = 5


def validate_answers() -> list[str]:
    """Validate answers.json against the schema and against questions.json's id set.

    :return: list of human-readable error messages; empty if everything passed
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        answers = json.load(f)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    errors = [
        f"{e.json_path}: {e.message}"
        for e in jsonschema.Draft202012Validator(schema).iter_errors(answers)
    ]

    question_ids = {q["id"] for q in questions}
    answer_ids = {a["id"] for a in answers} if isinstance(answers, list) else set()
    if answer_ids != question_ids:
        missing = question_ids - answer_ids
        extra = answer_ids - question_ids
        if missing:
            errors.append(f"answers.json is missing ids from questions.json: {sorted(missing)}")
        if extra:
            errors.append(f"answers.json has ids not in questions.json: {sorted(extra)}")

    return errors


# What it means for a sentence to actually state each fact. These are
# deliberately stricter than the concept triggers used during extraction: a
# recap saying a team "never trailed" mentions the concept without stating a
# quantity, and is supposed to match nothing. Only sentences that state the
# number, or name a go-ahead scorer, are owed a pattern.
STATED_FACT = {
    "deficit": (r"(\d+)[-\s]point\s+(?:deficit|hole)"
                r"|(?:trailed|trailing|down)\s+by\s+(?:\w+\s+){0,3}?\d+\s+points?"),
    "lead": (r"(?:lead|advantage|cushion|edge)[^.]{0,24}?"
             r"as\s+(?:many|much|big|large)\s+as\s+\d+"
             r"|ahead by\s+\d+\s+points|lead to\s+\d+"),
    "go-ahead": r"go-ahead|game[- ]winner|winning (?:basket|bucket|shot|score)",
}
MAX_UNMATCHED_SHOWN = 3


def validate_extraction_coverage() -> list[str]:
    """Check every recap sentence that states a fact against the patterns.

    A sentence the patterns cannot read is not an error in itself -- the model
    fallback exists for exactly that -- but it is a silent one: the fallback
    only runs when no pattern matched anywhere in the candidate set, so one
    unreadable phrasing among many readable ones simply lowers the maximum.
    That is how a 14-point comeback became a 13-point one.

    :return: list of human-readable error messages; empty if everything passed
    """
    from backend.config import DB_DSN
    from backend.extract import (DEFICIT_PATTERNS, GO_AHEAD_PATTERNS,
                                 LEAD_PATTERNS, split_sentences)

    patterns = {"deficit": DEFICIT_PATTERNS, "lead": LEAD_PATTERNS,
                "go-ahead": GO_AHEAD_PATTERNS}
    try:
        engine = sa.create_engine(DB_DSN)
        with engine.begin() as cx:
            bodies = cx.execute(
                sa.text("SELECT text FROM game_recaps ORDER BY recap_id")).scalars().all()
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        return [f"could not read game_recaps to check extraction coverage: {exc}"]

    errors = []
    for concept, stated in STATED_FACT.items():
        unmatched = [
            sentence
            for body in bodies
            for sentence in split_sentences(body)
            if re.search(stated, sentence, re.IGNORECASE)
            and not any(re.search(p, sentence, re.IGNORECASE) for p in patterns[concept])
        ]
        if unmatched:
            errors.append(
                f"{len(unmatched)} recap sentence(s) state a {concept} that no pattern in "
                f"extract.py reads, e.g.: " + " | ".join(s[:120] for s in
                                                         unmatched[:MAX_UNMATCHED_SHOWN]))
    return errors


def main() -> None:
    """Run validate_answers() and print PASS/FAIL, exiting 1 on failure."""
    errors = validate_answers() + validate_extraction_coverage()
    if not errors:
        print("PASS: questions/answers.json is schema-valid, its ids match "
              "questions/questions.json, and every stated deficit, lead and "
              "go-ahead in the recaps is readable.")
        return
    print(f"FAIL: {len(errors)} error(s) found.")
    for msg in errors[:MAX_ERRORS_SHOWN]:
        print(f"  - {msg}")
    if len(errors) > MAX_ERRORS_SHOWN:
        print(f"  ... and {len(errors) - MAX_ERRORS_SHOWN} more")
    sys.exit(1)


if __name__ == "__main__":
    main()
