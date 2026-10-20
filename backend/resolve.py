"""Deterministic entity resolution: question text -> table ids.

Every name, team, and date a question mentions is matched against the tables
here, before the LLM is involved at all. Nothing in this module is fuzzy: there
is no edit distance, no nearest-neighbour, no embedding. That is deliberate and
it is the single most important correctness property in the pipeline.

The data contains adversarial near-duplicate names -- "Darian Hadderley" and
"Dorian Hadderley" are both forwards on the same team -- so a one-character-tolerant
match would silently answer a question about one player using the other player's
stat line. An exact match that finds nothing is recoverable (the caller
abstains); a fuzzy match that finds the wrong person is not.

The other rule here is that resolution returns *candidate sets*, never a single
guess. A last name like "Hadderley" matches eight players; picking one of them at
this stage would be a coin flip. Ambiguity is handed to the caller, which either
narrows it with game context (see extract.resolve_name_in_game) or abstains.
"""
import re
from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Connection

# The season covered by the data, used for "this season" and as the default
# bound when a question gives only a partial date. Derived from game_details at
# load time rather than hard-coded, so a different season's data still works.
SEASON_START_FALLBACK = date(2025, 10, 21)
SEASON_END_FALLBACK = date(2026, 4, 12)

# README: "The All-Star break is February 13-18, 2026; 'after the break' means on
# or after February 19."
BREAK_START = date(2026, 2, 13)
BREAK_END = date(2026, 2, 18)
AFTER_BREAK_START = date(2026, 2, 19)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # Abbreviations, because questions write dates however people write them.
    # "Jan 18, 2026" matched nothing before this and the question abstained on a
    # game that was right there in the table.
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Longest first, so "january" is tried before "jan" and the alternation does not
# stop at the shorter prefix.
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

# Last day of each month; February is corrected for leap years in _month_range.
_MONTH_LAST_DAY = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                   7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def match_position(needle: str, haystack: str) -> int | None:
    """Return where `needle` first appears in `haystack`, or None.

    Used to order the teams a question names by where they are mentioned. In
    these questions the subject team is named first and the opponent second
    ("the Outriders' game against the Coyotes"), which is what lets the pipeline
    tell "who beat them" from "who they beat" without asking the model.

    :param needle: literal string to find (escaped, not a pattern)
    :param haystack: text to search
    :return: character offset of the first whole-word match, or None
    """
    if not needle:
        return None
    pattern = rf"(?<![\w-]){re.escape(needle)}(?![\w-])"
    m = re.search(pattern, haystack, re.IGNORECASE)
    return m.start() if m else None


def whole_word_match(needle: str, haystack: str) -> bool:
    """Case-insensitive whole-word/phrase search.

    Shared by team and player matching so both use identical semantics.

    Uses lookaround rather than \\b, and excludes hyphens from the boundary, so
    "Castell" does not match inside "Castell-Rooney" -- those are different people
    and a partial match would resolve to the wrong player.

    Apostrophes are deliberately *not* excluded, because the questions are full
    of possessives: "the Outriders' season opener" must still match the team
    "Outriders". Treating the trailing apostrophe as part of the word made every
    possessive reference resolve to no team at all.

    :param needle: literal string to find (not a pattern; it is escaped)
    :param haystack: text to search
    :return: True if needle appears in haystack delimited by non-word characters
    """
    if not needle:
        return False
    pattern = rf"(?<![\w-]){re.escape(needle)}(?![\w-])"
    return re.search(pattern, haystack, re.IGNORECASE) is not None


def load_teams(cx: Connection) -> list[dict]:
    """Load every team with the label fields used for matching and display.

    :param cx: open SQLAlchemy connection
    :return: one dict per team with team_id, city, name, abbreviation, conference
    """
    sql = "SELECT team_id, city, name, abbreviation, conference FROM teams ORDER BY team_id"
    return [dict(r) for r in cx.execute(text(sql)).mappings().all()]


def load_players(cx: Connection) -> list[dict]:
    """Load every player with a precomputed canonical "First Last" full name.

    players.team_id is nullable in the data (players not on a roster), so callers
    must not rely on it to decide who played for whom in a given game -- that
    comes from player_box_scores.team_id instead.

    :param cx: open SQLAlchemy connection
    :return: one dict per player with player_id, first_name, last_name, full_name, team_id
    """
    sql = "SELECT player_id, first_name, last_name, team_id FROM players ORDER BY player_id"
    rows = [dict(r) for r in cx.execute(text(sql)).mappings().all()]
    for r in rows:
        r["full_name"] = f"{r['first_name']} {r['last_name']}"
        # ingest.py lets pandas infer types, and a nullable integer column lands
        # as double precision, so team_id arrives as a float (or None).
        r["team_id"] = None if r["team_id"] is None else int(r["team_id"])
    return rows


def load_season_bounds(cx: Connection) -> tuple[date, date]:
    """Read the first and last game date in the data, for "this season" ranges.

    Read from the table rather than hard-coded so the pipeline still works on a
    different season's data.

    ingest.py writes game_timestamp as text (pandas infers the CSV column as a
    string), so it is cast here -- and everywhere else a date comparison happens.
    See GAME_DATE_SQL in execute.py for the shared expression.

    :param cx: open SQLAlchemy connection
    :return: (first game date, last game date)
    """
    sql = ("SELECT MIN(game_timestamp::timestamp::date) AS lo, "
           "MAX(game_timestamp::timestamp::date) AS hi FROM game_details")
    row = cx.execute(text(sql)).mappings().first()
    if not row or row["lo"] is None:
        return SEASON_START_FALLBACK, SEASON_END_FALLBACK
    return row["lo"], row["hi"]


def resolve_teams(question: str, teams: list[dict]) -> list[dict]:
    """Find every team the question names.

    Matches the mascot ("Outriders"), the full "City Mascot" ("Sablecrest Outriders"),
    the city alone ("Sablecrest"), and the abbreviation ("SBC"). All four appear in
    the data's own prose, so all four have to resolve.

    Results are ordered by where each team is mentioned, so the first entry is
    the question's subject and the second its opponent.

    :param question: raw question text
    :param teams: teams from load_teams()
    :return: matching team dicts in mention order, each with a `pos` offset;
        may be empty or hold several
    """
    matched = []
    for team in teams:
        labels = [
            f"{team['city']} {team['name']}",
            team["name"],
            team["city"],
            team["abbreviation"],
        ]
        positions = [p for p in (match_position(label, question) for label in labels)
                     if p is not None]
        if positions:
            matched.append(dict(team, pos=min(positions)))
    return sorted(matched, key=lambda t: (t["pos"], t["team_id"]))


def loose_name(value: str) -> str:
    """Normalize a name so hyphens and apostrophes do not block a match.

    "Castell-Rooney" and "castell rooney" are the same person written two ways,
    and requiring the hyphen meant a question spelling it with a space resolved
    to five unrelated players called Castell instead.

    This is formatting, not fuzziness: no character is added, dropped or
    substituted, so "Deshawn" still does not match "Dashawn".

    :param value: a name or a stretch of question text
    :return: lowercased, with hyphens and apostrophes as spaces
    """
    return re.sub(r"\s+", " ", re.sub(r"[-'’]", " ", value.lower())).strip()


def resolve_players(question: str, players: list[dict]) -> list[dict]:
    """Find every player the question names, exactly.

    Two passes, in strict priority order:

    1. Full name ("Darian Hadderley"). If any full name matches, only full-name
       matches are returned -- a question that names a player completely has
       already disambiguated them, and falling through to the last-name pass
       would drag in every other Hadderley.
    2. Last name only ("Hadderley"), returning *all* players with that last name.

    A last-name-only result with more than one entry is ambiguous on purpose. The
    caller must narrow it with game context or abstain; this function will not
    guess.

    :param question: raw question text
    :param players: players from load_players()
    :return: matching player dicts in player_id order; `exact` is True when the
        match came from the full-name pass
    """
    full = [p for p in players if whole_word_match(p["full_name"], question)]
    if not full:
        # Retry with hyphens and apostrophes treated as spaces, so a question
        # written "Phineas Castell Rooney" still finds "Phineas Castell-Rooney".
        loose_q = loose_name(question)
        full = [p for p in players
                if whole_word_match(loose_name(p["full_name"]), loose_q)]
    if full:
        return [dict(p, exact=True) for p in full]
    last = [p for p in players if whole_word_match(p["last_name"], question)]
    return [dict(p, exact=False) for p in last]


def _month_range(year: int, month: int) -> tuple[date, date]:
    """Return the first and last calendar day of a month.

    :param year: four-digit year
    :param month: 1-12
    :return: (first day, last day), leap-year aware for February
    """
    last = _MONTH_LAST_DAY[month]
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        last = 29
    return date(year, month, 1), date(year, month, last)


def resolve_dates(question: str, season: tuple[date, date]) -> dict:
    """Extract a date range from the question.

    Handles, in priority order: an explicit single day ("April 12, 2026" or
    "2026-04-12"), a month and year ("January 2026"), and the All-Star break
    ("after the break"). Falls back to the full season range, which is a no-op
    filter rather than a guess.

    game_timestamp is naive local time and no game in the data tips off late
    enough to cross midnight, so the literal date is always the game date. This
    function never converts timezones, and neither should anything downstream.

    :param question: raw question text
    :param season: (first, last) game date from load_season_bounds()
    :return: {"date_from": date, "date_to": date, "exact_day": bool}
    """
    season_start, season_end = season
    lowered = question.lower()

    # "April 12, 2026" / "April 12 2026" / "12 April 2026"
    month_alt = MONTH_ALT
    m = re.search(rf"\b({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", lowered)
    if m:
        d = date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        return {"date_from": d, "date_to": d, "exact_day": True}
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_alt}),?\s+(\d{{4}})\b", lowered)
    if m:
        d = date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
        return {"date_from": d, "date_to": d, "exact_day": True}

    # ISO "2026-04-12"
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", question)
    if iso:
        d = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        return {"date_from": d, "date_to": d, "exact_day": True}

    # Numeric "3/12/2026" or "2-25-2026", read month-first. Unparsed, these fell
    # through to the whole-season range, and the question was then answered
    # about whichever game the rest of the filters happened to select.
    num = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", question)
    if num:
        month, day, year = int(num.group(1)), int(num.group(2)), int(num.group(3))
        # A first field above 12 can only be a day, so read that one day-first.
        if month > 12 and day <= 12:
            month, day = day, month
        try:
            return {"date_from": date(year, month, day),
                    "date_to": date(year, month, day), "exact_day": True}
        except ValueError:
            pass

    # A span of months: "between January and March 2026", "from October to
    # December 2025". Checked before the single-month pattern, which would
    # otherwise match only the second month and silently shrink the range to it.
    span = re.search(
        rf"\b(?:between|from)\s+({month_alt})(?:\s+(\d{{4}}))?\s+(?:and|to|through|-)\s+"
        rf"({month_alt})\s+(\d{{4}})\b", lowered)
    if span:
        end_year = int(span.group(4))
        start_year = int(span.group(2)) if span.group(2) else end_year
        start_month, end_month = MONTHS[span.group(1)], MONTHS[span.group(3)]
        # An unstated start year that sorts after the end month belongs to the
        # previous calendar year, since a season runs across the new year.
        if span.group(2) is None and start_month > end_month:
            start_year = end_year - 1
        lo, _ = _month_range(start_year, start_month)
        _, hi = _month_range(end_year, end_month)
        return {"date_from": lo, "date_to": hi, "exact_day": False}

    # "January 2026" (month + year, no day)
    my = re.search(rf"\b({month_alt})\s+(\d{{4}})\b", lowered)
    if my:
        lo, hi = _month_range(int(my.group(2)), MONTHS[my.group(1)])
        return {"date_from": lo, "date_to": hi, "exact_day": False}

    # A month and day with the year left unsaid: "on January 23". People drop
    # the year constantly, and with no pattern for it the range stayed at the
    # full season -- "who was injured on january 23" was answered about the
    # whole year, naming the player with the most absences in it.
    #
    # Checked after the patterns above so that "January 2026" is read as a
    # month and "January 23, 2026" as a full date; the word boundary on the day
    # is what stops "2026" being read as the 20th.
    md = re.search(rf"\b({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lowered)
    if md:
        month, day = MONTHS[md.group(1)], int(md.group(2))
    else:
        md = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_alt})\b", lowered)
        month, day = (MONTHS[md.group(2)], int(md.group(1))) if md else (None, None)
    if month is not None:
        # A season runs across the new year, so the year is whichever of the two
        # puts the date inside it. A date that lands in neither is left in the
        # season's opening year, where it matches no games and the question is
        # abstained on rather than widened back out to the whole season.
        candidates = []
        for year in (season_start.year, season_end.year):
            try:
                candidates.append(date(year, month, day))
            except ValueError:
                pass
        if candidates:
            inside = [d for d in candidates if season_start <= d <= season_end]
            picked = inside[0] if inside else candidates[0]
            return {"date_from": picked, "date_to": picked, "exact_day": True}

    # "The 2025 season" names the whole season, not the calendar year. Every
    # game in this data carries season 2025 while spanning two calendar years,
    # so the phrase has to be read before the bare year below.
    season_phrase = re.search(r"\b(?:the\s+)?\d{4}(?:\s*[-/]\s*\d{2,4})?\s+season\b", lowered)
    if season_phrase:
        return {"date_from": season_start, "date_to": season_end, "exact_day": False}

    # A bare year: "who scored the most points in 2025". The season runs across
    # the new year, so a calendar year is only part of it -- and with no year
    # filter at all, "in 2025" and "in 2026" both returned the full-season
    # leader, giving the same answer to two different questions.
    year = re.search(r"\b(20\d{2})\b", question)
    if year:
        value = int(year.group(1))
        lo = max(date(value, 1, 1), season_start)
        hi = min(date(value, 12, 31), season_end)
        # A year outside the season leaves lo after hi, which matches no games
        # and makes the caller abstain rather than answer about another year.
        return {"date_from": lo, "date_to": hi, "exact_day": False}

    # All-Star break, relative to the fixed break dates in the README.
    if "break" in lowered:
        if re.search(r"\b(after|since|following|post)\b", lowered):
            return {"date_from": AFTER_BREAK_START, "date_to": season_end, "exact_day": False}
        if re.search(r"\b(before|prior to|pre)\b", lowered):
            return {"date_from": season_start, "date_to": BREAK_START, "exact_day": False}

    return {"date_from": season_start, "date_to": season_end, "exact_day": False}


def disambiguate_person(cx: Connection, candidate_ids: list[int], game_ids: list[int],
                        team_id: int | None = None,
                        exclude_team_id: int | None = None,
                        source: str = "player_box_scores") -> int | None:
    """Narrow a set of same-surname players down to the one the question means.

    A bare surname is ambiguous against the whole league -- twelve players are
    named Hadderley -- but rarely ambiguous inside the games a question is actually
    about. Only one Hadderley appears in the Outriders' season opener, so "how many
    points did Hadderley score in the Outriders' season opener" has exactly one
    answer, and refusing it would be needlessly timid. Measured over the season,
    87% of (surname, game) pairs are unique within the game alone and another 6%
    are settled by the team the question names; only ~6% are truly ambiguous.

    Tried in order, widening only when the narrower attempt finds nobody:

    1. Candidates who appear in these games *on the named team*. This is what
       separates two players who share a surname across opposing benches.
    2. Candidates who appear in these games at all -- the fallback for when the
       team named in the question is the opponent rather than the player's own.
    3. Nothing. Two players with the same surname on the same team in the same
       game cannot be told apart from the question, so the caller abstains.

    This mirrors extract.resolve_name_in_game, which does the same job for names
    lifted out of recap prose; the difference is only where the name came from.

    :param cx: open SQLAlchemy connection
    :param candidate_ids: player_ids sharing the surname the question used
    :param game_ids: the games the plan's filters already narrowed to
    :param team_id: the team the question named, if the player is on it
    :param exclude_team_id: the team the player is playing *against*, for
        questions phrased as "X scored against the Outriders"
    :param source: the table the plan reads, so an injury question is settled
        against the notes rather than the box scores
    :return: the single matching player_id, or None if it stays ambiguous
    """
    if not candidate_ids or not game_ids:
        return None
    params = {"pids": [int(p) for p in candidate_ids],
              "gids": [int(g) for g in game_ids]}
    # An injury question has to be settled against the notes, not the box
    # scores: a player who was out has no box score row that night, so looking
    # there finds nobody and the surname stays ambiguous. "Why was Novak
    # injured" then ran with no player filter at all and read every note in the
    # season.
    table, column = (("injury_notes", "player_id") if source == "injury_notes"
                     else ("player_box_scores", "person_id"))
    base = (f"SELECT DISTINCT {column} AS person_id FROM {table} "
            f"WHERE {column} = ANY(:pids) AND game_id = ANY(:gids)")

    if team_id is not None:
        rows = cx.execute(text(base + " AND team_id = :tid ORDER BY person_id"),
                          dict(params, tid=int(team_id))).scalars().all()
        if len(rows) == 1:
            return int(rows[0])

    if exclude_team_id is not None:
        rows = cx.execute(text(base + " AND team_id <> :xtid ORDER BY person_id"),
                          dict(params, xtid=int(exclude_team_id))).scalars().all()
        if len(rows) == 1:
            return int(rows[0])

    rows = cx.execute(text(base + " ORDER BY person_id"), params).scalars().all()
    return int(rows[0]) if len(rows) == 1 else None


# Capitalized words that are not somebody's name. Question openers, calendar
# words, and the league vocabulary that shows up capitalized in these questions.
PROPER_NOUN_STOPWORDS = {
    "who", "what", "which", "when", "where", "why", "how", "did", "do", "does",
    "was", "were", "is", "are", "has", "have", "had", "the", "a", "an", "and",
    "or", "of", "for", "to", "by", "with", "against", "versus", "vs", "after",
    "before", "during", "in", "on", "at", "this", "that", "their", "his", "her",
    "its", "they", "them", "he", "she", "it", "many", "much", "most", "least",
    "all", "star", "all-star", "conference", "east", "west", "eastern",
    "western", "north", "south", "northern", "southern", "division", "season",
    "game", "games", "points", "team", "teams", "player", "players", "nba",
    "final", "score", "break", "home", "road", "away", "overtime", "quarter",
    "half", "halftime", "since", "between", "from", "per", "average", "total",
    "no", "not", "if", "as", "than", "then", "there", "his",
    # Words that open a question and are therefore capitalized by grammar
    # rather than because they name anything. "Across games Bramport won by at
    # least 15..." was refused outright because "Across" matched no team.
    "across", "among", "amongst", "within", "throughout", "going", "heading",
    "including", "excluding", "regarding", "concerning", "per", "via",
    "name", "list", "show", "tell", "give", "find", "compare", "count",
    "total", "can", "could", "would", "should", "will", "do", "done",
    "over", "under", "about", "into", "onto", "upon", "out", "up", "down",
    "when", "while", "whose", "whom", "only", "just", "also", "both",
    "each", "every", "any", "some", "more", "less", "fewer", "most",
    "least", "best", "worst", "top", "bottom", "first", "last", "next",
    "same", "other", "another", "such", "what's", "whats", "who's",
}
PROPER_NOUN_STOPWORDS |= {m for m in MONTHS}


def unresolved_proper_nouns(question: str, resolved: dict) -> list[str]:
    """Find capitalized names in a question that matched nothing in the data.

    This is the guard against the pipeline's most dangerous failure: a filter
    that cannot be built is simply not applied, and the query then runs across
    everything. Asked about a player who does not exist, an earlier version
    returned 194,761 points -- every point scored in the league that season --
    with `answerable: true`. Asked about a team that does not exist, it counted
    all 841 games. Nothing downstream could detect either, because both answers
    were correctly computed from real rows; only the question was wrong.

    So a name the data does not contain has to be noticed here. Capitalized
    words are collected into phrases, anything covered by a team or player that
    *did* resolve is removed, calendar and league vocabulary is ignored, and
    whatever is left is a name nobody recognises -- which makes the question
    unanswerable rather than unconstrained.

    :param question: raw question text
    :param resolved: output of resolve_question()
    :return: unrecognized capitalized phrases, empty when everything resolved
    """
    covered = []
    for team in resolved["teams"]:
        covered += [f"{team['city']} {team['name']}", team["name"], team["city"],
                    team["abbreviation"]]
    for player in resolved["players"]:
        covered += [player["full_name"], player["first_name"], player["last_name"]]

    leftovers = []
    # Consecutive capitalized tokens form one phrase: "Zephyr Quillfeather" is
    # one name, not two unknowns.
    for phrase in re.findall(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*", question):
        words = phrase.split()
        # Drop the leading word when it opens the sentence, since English
        # capitalizes it regardless ("Which team ...").
        if question.lstrip().startswith(phrase) and words[0].lower() in PROPER_NOUN_STOPWORDS:
            words = words[1:]
        remaining = [w for w in words if w.lower().strip(".,?!") not in PROPER_NOUN_STOPWORDS]
        if not remaining:
            continue
        candidate = " ".join(remaining)
        if any(whole_word_match(c, candidate) or whole_word_match(candidate, c)
               for c in covered if c):
            continue
        leftovers.append(candidate)
    return leftovers


# Where a question puts the thing it is asking about: "how many points did
# <SUBJECT> score", "what did <SUBJECT> finish with". Capturing this slot is how
# an unrecognized name gets noticed even when the question is written in
# lowercase, which the capitalized-word scan above cannot see at all.
SUBJECT_SLOT = (r"\bdid\s+(.{1,48}?)\s+"
                r"(?:score|scored|have|had|put\s+up|get|got|make|made|shoot|shot"
                r"|play|played|average|averaged|finish|finished|record|recorded"
                r"|tally|tallied|miss|missed|win|won|lose|lost|grab|grabbed"
                r"|dish|dished|pull|pulled|attempt|attempted)\b")

# Words inside a subject phrase that never name anybody.
SUBJECT_STOPWORDS = {"the", "a", "an", "his", "her", "their", "its", "this",
                     "that", "these", "those", "team", "player", "any", "all",
                     "each", "both", "of", "in", "on", "at", "for", "to",
                     # Pronouns refer back to something already resolved, so
                     # they name nobody new. "...and how many points did he
                     # score?" would otherwise read "he" as an unknown person
                     # and abstain on a question it had already answered.
                     "he", "she", "they", "it", "him", "them", "we", "you",
                     "who", "whom", "someone", "anyone", "everyone", "nobody",
                     "guy", "one", "others"}


def unresolved_subject(question: str, resolved: dict) -> str | None:
    """Return the question's subject when it matched no team or player.

    The capitalized-word guard cannot see a lowercase name, and a name that
    resolves to nobody produces no filter at all -- so "how many points did
    damian sinclar score in 2025" summed every player in the league and
    reported 74,659 as one man's total. Reading the grammatical slot instead of
    the capitalization catches it however the question is typed.

    :param question: raw question text
    :param resolved: output of resolve_question()
    :return: the unrecognized subject phrase, or None when it resolved
    """
    m = re.search(SUBJECT_SLOT, question, re.IGNORECASE)
    if not m:
        return None
    phrase = m.group(1).strip()
    tokens = [t for t in re.split(r"[\s,]+", phrase)
              if t and t.lower().strip(".'’") not in SUBJECT_STOPWORDS]
    if not tokens:
        return None

    covered = []
    for team in resolved["teams"]:
        covered += [team["name"], team["city"], team["abbreviation"]]
    for player in resolved["players"]:
        covered += [player["first_name"], player["last_name"], player["full_name"]]

    loose_covered = " ".join(loose_name(c) for c in covered if c)
    if any(whole_word_match(loose_name(t), loose_covered) for t in tokens):
        return None
    return phrase


def resolve_question(question: str, teams: list[dict], players: list[dict],
                     season: tuple[date, date]) -> dict:
    """Resolve every entity in one question.

    :param question: raw question text
    :param teams: teams from load_teams()
    :param players: players from load_players()
    :param season: bounds from load_season_bounds()
    :return: {"teams": [...], "players": [...], "dates": {...}}
    """
    return {
        "teams": resolve_teams(question, teams),
        "players": resolve_players(question, players),
        "dates": resolve_dates(question, season),
    }
