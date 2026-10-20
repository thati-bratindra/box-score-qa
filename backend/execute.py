"""Execute a query plan against the tables. No LLM, no guessing.

Everything that has to be *correct* lives here: which rows qualify, what the
totals are, which game is "first", and which rows get cited. The model chose the
shape of the query in plan.py; this module is what actually answers it.

Two properties matter more than anything else in this file:

**Evidence is a by-product, not a claim.** The executor records every row it
touches while computing, and those rows become the citation list. Nothing else
can cite a row. This removes a whole failure mode -- a model naming a real row
next to a wrong number -- because the number is derived from exactly the rows
that get cited.

**Determinism.** A fresh run of the pipeline must match the committed
answers.json, so every ORDER BY carries an explicit id tie-breaker and every
aggregate resolves ties by a fixed rule. Two runs over the same data must return
byte-identical output.
"""
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.query import select_rows

# ingest.py stores game_timestamp as text (pandas types the CSV column as a
# string), so every date comparison has to cast. Kept in one place so the cast
# can never drift between queries.
#
# The cast is to a naive timestamp on purpose. No game in the data tips off late
# enough to cross midnight, so the literal date is always the game date;
# converting to UTC first would shift evening games onto the following day and
# quietly break every date filter.
GAME_DATE_SQL = "g.game_timestamp::timestamp::date"
GAME_ORDER_SQL = "g.game_timestamp::timestamp, g.game_id"

# Derived columns the plan may aggregate over that have no single database
# column behind them.
DERIVED_BOX_FIELDS = {"rebounds": ("offensive_reb", "defensive_reb")}

# Shooting percentages, as (makes, attempts) column pairs. These are ratios, so
# combining them across games means summing both halves and dividing once --
# never averaging the per-game percentages, which would weight a single 1-for-1
# night equally with a 10-for-20 one.
RATIO_BOX_FIELDS = {
    "fg3_pct": (("fg3_made",), ("fg3_attempted",)),
    "ft_pct": (("ft_made",), ("ft_attempted",)),
    "fg2_pct": (("fg2_made",), ("fg2_attempted",)),
    "fg_pct": (("fg2_made", "fg3_made"), ("fg2_attempted", "fg3_attempted")),
}

# A percentage over a handful of attempts is noise, and "best three-point
# shooter" asking over one made three is not what anyone means. Grouped
# percentage comparisons require this many attempts to qualify; a player below
# it is excluded rather than allowed to win on a tiny sample.
MIN_ATTEMPTS_FOR_RATIO = 10


class ExecResult:
    """The outcome of running one plan.

    :ivar values: primitive name -> computed value, consumed by plan `outputs`
    :ivar games: game_details rows touched, keyed by game_id
    :ivar box: (game_id, person_id) pairs touched
    :ivar recaps: recap_ids touched
    :ivar notes: note_ids touched
    :ivar empty: True when no qualifying rows were found, which is the signal to
        abstain rather than to report a zero
    """

    def __init__(self):
        self.values: dict = {}
        self.games: dict[int, dict] = {}
        self.box: set[tuple[int, int]] = set()
        self.recaps: set[int] = set()
        self.notes: set[int] = set()
        self.empty = False

    def touch_game(self, row: dict) -> None:
        """Record a game_details row as evidence.

        :param row: a row from select_games()
        """
        self.games[int(row["game_id"])] = row

    def touch_box(self, row: dict) -> None:
        """Record a player_box_scores row, and implicitly its parent game.

        Citing a box score without the game it came from would leave questions
        that declare both tables ungrounded, so the pairing is enforced here
        rather than left to each call site.

        :param row: a row from select_box_rows()
        """
        self.box.add((int(row["game_id"]), int(row["person_id"])))


def box_field(row: dict, field: str) -> float:
    """Read an aggregate field off a box score row, including derived ones.

    :param row: a player_box_scores row
    :param field: column name, or a derived name such as "rebounds"
    :return: the numeric value, 0 when the column is absent or null
    """
    if field in DERIVED_BOX_FIELDS:
        return sum(float(row.get(c) or 0) for c in DERIVED_BOX_FIELDS[field])
    if field in RATIO_BOX_FIELDS:
        made, attempted = ratio_parts([row], field)
        return (made / attempted) if attempted else 0.0
    return float(row.get(field) or 0)


def ratio_parts(rows: list[dict], field: str) -> tuple[float, float]:
    """Total the makes and attempts behind a percentage field.

    :param rows: box score rows to combine
    :param field: a key of RATIO_BOX_FIELDS
    :return: (total makes, total attempts)
    """
    made_cols, attempt_cols = RATIO_BOX_FIELDS[field]
    made = sum(float(r.get(c) or 0) for r in rows for c in made_cols)
    attempted = sum(float(r.get(c) or 0) for r in rows for c in attempt_cols)
    return made, attempted


def game_predicates(filters: dict) -> list[dict]:
    """Translate the plan's named game filters into generic predicates.

    The named keys are kept as the surface the inference layer writes to, but
    they are sugar: every one of them is a predicate over a registered column or
    one of the team-relative relations. A new filter is a line here plus a
    column in the registry, rather than a new branch in the query builder.

    :param filters: plan["game_filters"]
    :return: predicate list for query.select_rows
    :raises ValueError: if a team-relative filter is given with no team
    """
    preds: list[dict] = []
    team = filters.get("team_id")
    if team is not None:
        preds.append({"op": "played_in", "val": int(team)})
    if filters.get("opponent_team_id") is not None:
        preds.append({"op": "played_in", "val": int(filters["opponent_team_id"])})
    if filters.get("game_id") is not None:
        preds.append({"col": "game_id", "op": "=", "val": int(filters["game_id"])})
    if filters.get("date_from"):
        preds.append({"col": "game_date", "op": ">=", "val": str(filters["date_from"])})
    if filters.get("date_to"):
        preds.append({"col": "game_date", "op": "<=", "val": str(filters["date_to"])})
    if filters.get("min_margin") is not None:
        preds.append({"col": "margin", "op": ">=", "val": int(filters["min_margin"])})
    if filters.get("max_margin") is not None:
        preds.append({"col": "margin", "op": "<=", "val": int(filters["max_margin"])})

    # Outcome, venue and conference are all "relative to the subject team", so
    # each needs one. Without a team there is no side to be on, and a filter
    # that cannot be applied must not be silently dropped.
    outcome = filters.get("outcome")
    if outcome in ("win", "loss"):
        if team is None:
            raise ValueError("outcome filter needs a team")
        preds.append({"op": "won" if outcome == "win" else "lost", "val": int(team)})

    venue = filters.get("venue")
    if venue in ("home", "away"):
        if team is None:
            raise ValueError("venue filter needs a team")
        preds.append({"op": "hosted" if venue == "home" else "visited", "val": int(team)})

    conference = filters.get("opponent_conference")
    if conference in ("East", "West"):
        if team is None:
            raise ValueError("conference filter needs a team")
        preds.append({"op": "opponent_conference", "val": (int(team), conference)})

    if filters.get("exclude_person_id") is not None:
        preds.append({"op": "absent", "val": int(filters["exclude_person_id"])})
    preds += list(filters.get("where") or [])
    return preds


def row_predicates(filters: dict, game_ids: list[int]) -> list[dict]:
    """Translate the plan's named row filters into generic predicates.

    :param filters: plan["row_filters"]
    :param game_ids: games already narrowed by select_games()
    :return: predicate list for query.select_rows
    """
    preds: list[dict] = [{"col": "game_id", "op": "in", "val": [int(g) for g in game_ids]}]
    simple = (("person_id", "person_id", "="), ("team_id", "team_id", "="),
              ("exclude_team_id", "team_id", "!="), ("starter", "starter", "="),
              ("min_points", "points", ">="), ("max_points", "points", "<="),
              ("min_rebounds", "rebounds", ">="), ("min_assists", "assists", ">="))
    for key, col, op in simple:
        if filters.get(key) is not None:
            value = filters[key]
            preds.append({"col": col, "op": op,
                          "val": bool(value) if col == "starter" else int(value)})
    if filters.get("person_ids"):
        preds.append({"col": "person_id", "op": "in",
                      "val": [int(p) for p in filters["person_ids"]]})
    if filters.get("position"):
        preds.append({"col": "position", "op": "like", "val": f"%{filters['position']}%"})
    preds += list(filters.get("where") or [])
    return preds


def select_games(cx: Connection, filters: dict, select: str) -> list[dict]:
    """Find the games a plan's game_filters describe, in stable date order.

    :param cx: open SQLAlchemy connection
    :param filters: plan["game_filters"]
    :param select: "first", "last", "all", or an integer position
    :return: game rows, earliest first, with a derived margin and game_date
    """
    try:
        preds = game_predicates(filters)
    except ValueError:
        # A team-relative filter with no team is malformed. Returning no games
        # makes the caller abstain, which is far safer than dropping the
        # condition and aggregating over every game in the range.
        return []
    rows = select_rows(cx, "game_details", preds)

    if not rows:
        return []
    if select == "first":
        return rows[:1]
    if select == "last":
        return rows[-1:]
    # A positional selection: "their 10th game" is the 10th in schedule order.
    # Out of range returns nothing, which makes the caller abstain rather than
    # silently fall back to the whole list.
    if isinstance(select, int):
        return rows[select - 1:select] if 1 <= select <= len(rows) else []
    return rows


def select_box_rows(cx: Connection, game_ids: list[int], filters: dict) -> list[dict]:
    """Fetch player box score rows inside a set of games.

    :param cx: open SQLAlchemy connection
    :param game_ids: games already narrowed by select_games()
    :param filters: plan["row_filters"]
    :return: box score rows ordered by (game_id, person_id) for determinism
    """
    if not game_ids:
        return []
    rows = select_rows(cx, "player_box_scores", row_predicates(filters, game_ids))

    # Double- and triple-doubles are "two (or three) categories in double
    # figures", which counts across columns rather than comparing one of them,
    # so it is applied here rather than as a predicate.
    if filters.get("double_double") or filters.get("triple_double"):
        needed = 3 if filters.get("triple_double") else 2
        rows = [r for r in rows if double_figure_count(r) >= needed]
    return rows


def double_figure_count(row: dict) -> int:
    """Count how many statistical categories reached double figures.

    :param row: a player_box_scores row
    :return: how many of points, rebounds, assists, steals and blocks are >= 10
    """
    categories = (box_field(row, "points"), box_field(row, "rebounds"),
                  box_field(row, "assists"), box_field(row, "steals"),
                  box_field(row, "blocks"))
    return sum(1 for value in categories if value >= 10)


def format_score(game: dict) -> str:
    """Format a final score as "<winner points>-<loser points>".

    Built from the table, never from model output, so the README's format rule
    holds regardless of what the LLM wrote.

    :param game: a game row from select_games()
    :return: e.g. "117-102"
    """
    high = max(int(game["home_points"]), int(game["away_points"]))
    low = min(int(game["home_points"]), int(game["away_points"]))
    return f"{high}-{low}"


def opponent_of(game: dict, team_id: int | None) -> int | None:
    """Identify the other team in a game.

    :param game: a game row from select_games()
    :param team_id: the subject team, or None if the question named no subject
    :return: the opposing team_id, or None when it cannot be determined
    """
    if team_id is None:
        return None
    if int(game["home_team_id"]) == int(team_id):
        return int(game["away_team_id"])
    if int(game["away_team_id"]) == int(team_id):
        return int(game["home_team_id"])
    return None


def nth_value(values: list[float], rank: int, descending: bool) -> float | None:
    """Return the rank-th *distinct* value, for "second-highest"-style questions.

    Ranking is over distinct values, which is what the phrase means: if two
    players tie for the lead, the second-highest score is the next score down,
    not the other leader. Returns None when the data does not go that deep,
    which makes the caller abstain rather than report the last place it found.

    :param values: the values to rank
    :param rank: 1 for the top value, 2 for the next distinct one, and so on
    :param descending: True for highest-first, False for lowest-first
    :return: the value at that rank, or None if there are fewer distinct values
    """
    ordered = sorted(set(values), reverse=descending)
    index = max(1, rank) - 1
    return ordered[index] if index < len(ordered) else None


def _group_rows(rows: list[dict], field: str, inner: str) -> dict[int, float]:
    """Combine box score rows into one value per player.

    :param rows: box score rows
    :param field: the stat being aggregated
    :param inner: "sum" to total the stat, "count" to count qualifying rows,
        "max" for the player's single best game
    :return: person_id -> combined value
    """
    # Percentages combine as one ratio of totals per player, and players with
    # too few attempts are dropped rather than allowed to top the list on a
    # single make.
    if field in RATIO_BOX_FIELDS:
        by_player: dict[int, list[dict]] = {}
        for row in rows:
            by_player.setdefault(int(row["person_id"]), []).append(row)
        ratios = {}
        for pid, player_rows in by_player.items():
            made, attempted = ratio_parts(player_rows, field)
            if attempted >= MIN_ATTEMPTS_FOR_RATIO:
                ratios[pid] = made / attempted
        return ratios

    grouped: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        pid = int(row["person_id"])
        value = box_field(row, field)
        counts[pid] = counts.get(pid, 0) + 1
        if inner == "count":
            grouped[pid] = counts[pid]
        elif inner == "max":
            grouped[pid] = max(grouped.get(pid, value), value)
        elif inner == "min":
            # A player's quietest game, for "the fewest points in a game".
            grouped[pid] = min(grouped.get(pid, value), value)
        else:
            grouped[pid] = grouped.get(pid, 0) + value
    if inner == "avg":
        return {pid: total / counts[pid] for pid, total in grouped.items()}
    return grouped


def tied_names(tied: list[int], players: dict[int, dict]) -> str:
    """Name every player a ranking ended in a tie between.

    Reporting only the first of them stated as a fact something the data does
    not say. It also silently turned set questions into top-one ones: "who was
    injured on January 23" is a tie between everyone listed that night, and
    answering with one of the two dozen names reads as though the rest were not
    injured. The ids are sorted first, so repeated runs agree.

    :param tied: player_ids sharing the winning value, already sorted
    :param players: player_id -> player dict
    :return: the names, comma-separated, in player_id order
    """
    named = [players.get(p, {}).get("full_name", "") for p in tied]
    return ", ".join(n for n in named if n)


def _apply_argmax(result: ExecResult, rows: list[dict], agg: dict,
                  players: dict[int, dict]) -> list[dict]:
    """Resolve an argmax/argmin over players and record the contributing rows.

    On a tie the winner is the lowest person_id, so repeated runs agree, and the
    rows of *every* tied player are cited -- the evidence should show the tie
    rather than hide it behind an arbitrary pick.

    :param result: ExecResult to write values and evidence into
    :param rows: qualifying box score rows
    :param agg: plan["aggregate"]
    :param players: player_id -> player dict, for the winner's name
    :return: the rows that contributed to the reported value
    """
    grouped = _group_rows(rows, agg["field"], agg["inner"])
    if not grouped:
        return []

    best = nth_value(list(grouped.values()), agg.get("rank", 1), agg["op"] == "argmax")
    if best is None:
        return []
    tied = sorted(pid for pid, value in grouped.items() if value == best)

    result.values["agg"] = best
    result.values["group_name"] = tied_names(tied, players)
    # Cite every tied player, not just the one reported.
    return [r for r in rows if int(r["person_id"]) in set(tied)]


def _max_streak(cx: Connection, plan: dict, result: ExecResult) -> ExecResult:
    """Find the longest run of consecutive wins or losses.

    The outcome filter is deliberately removed before selecting, because a
    streak is defined by the games *between* the ones that qualify. Filtering to
    wins first leaves a list with no losses in it, at which point every game
    looks consecutive and the whole season reads as one streak.

    :param cx: open SQLAlchemy connection
    :param plan: the plan being executed
    :param result: ExecResult to write into
    :return: the ExecResult, with the streak's games cited
    """
    filters = dict(plan["game_filters"])
    want_win = filters.pop("outcome", "win") != "loss"
    team = filters.get("team_id")
    if team is None:
        result.empty = True
        return result

    ordered = select_games(cx, filters, "all")
    if not ordered:
        result.empty = True
        return result

    best: list[dict] = []
    current: list[dict] = []
    for game in ordered:
        won = int(game["winning_team_id"]) == int(team)
        if won == want_win:
            current.append(game)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    if not best:
        result.empty = True
        return result

    for game in best:
        result.touch_game(game)
    result.values["agg"] = len(best)
    result.values["n_games"] = len(best)
    result.values["n_rows"] = len(best)
    result.values["subject_team"] = team
    return result


def _argmax_by_opponent(result: ExecResult, games: list[dict], agg: dict,
                        subject_team: int | None,
                        teams: dict[int, dict]) -> list[dict]:
    """Rank opponents by how often, or how heavily, they featured.

    Answers "which team did they beat most often" and its relatives. Ties break
    on the lowest team_id so a rerun agrees, and every game against the winning
    opponent is cited.

    :param result: ExecResult to write values into
    :param games: candidate games
    :param agg: plan["aggregate"]
    :param subject_team: the team the question is about
    :param teams: team_id -> team dict
    :return: the games contributing to the reported opponent
    """
    if subject_team is None:
        return []
    totals: dict[int, float] = {}
    by_opponent: dict[int, list[dict]] = {}
    for game in games:
        opponent = opponent_of(game, subject_team)
        if opponent is None:
            continue
        by_opponent.setdefault(opponent, []).append(game)
        step = 1.0 if agg["inner"] == "count" else float(game.get(agg["field"]) or 0)
        totals[opponent] = totals.get(opponent, 0) + step
    if not totals:
        return []

    best = nth_value(list(totals.values()), agg.get("rank", 1), agg["op"] == "argmax")
    if best is None:
        return []
    winner = min(t for t, v in totals.items() if v == best)
    result.values["agg"] = best
    result.values["group_name"] = teams.get(winner, {}).get("name", "")
    result.values["opponent_name"] = teams.get(winner, {}).get("name", "")
    return by_opponent[winner]


def count_notes(cx: Connection, game_ids: list[int], person_id: int | None,
                status: str | None = None) -> list[dict]:
    """Load injury notes for a set of games, for counting absences.

    :param cx: open SQLAlchemy connection
    :param game_ids: games in scope
    :param person_id: restrict to one player, or None
    :param status: restrict to a status such as "out", or None
    :return: matching note rows
    """
    if not game_ids:
        return []
    where = ["n.game_id = ANY(:gids)"]
    params: dict = {"gids": [int(g) for g in game_ids]}
    if person_id is not None:
        where.append("n.player_id = :pid")
        params["pid"] = int(person_id)
    if status is not None:
        where.append("n.status = :status")
        params["status"] = status
    sql = ("SELECT n.note_id, n.game_id, n.player_id, n.status FROM injury_notes n "
           "WHERE " + " AND ".join(where) + " ORDER BY n.game_id, n.note_id")
    return [dict(r) for r in cx.execute(text(sql), params).mappings().all()]


def execute_plan(cx: Connection, plan: dict, players: dict[int, dict],
                 teams: dict[int, dict]) -> ExecResult:
    """Run a validated plan and return its values plus the rows it touched.

    :param cx: open SQLAlchemy connection
    :param plan: a plan that has passed plan.validate_plan()
    :param players: player_id -> player dict (needs full_name)
    :param teams: team_id -> team dict (needs name)
    :return: an ExecResult; check `.empty` before trusting `.values`
    """
    result = ExecResult()
    agg = plan["aggregate"]

    if agg["op"] == "max_streak":
        return _max_streak(cx, plan, result)

    games = select_games(cx, plan["game_filters"], plan["game_select"])
    if not games:
        result.empty = True
        return result

    subject_team = plan["game_filters"].get("team_id") or plan["row_filters"].get("team_id")

    # A game-level max/min picks one game and reports facts about it, so it has
    # to be resolved before any box score work narrows the set.
    #
    # `focus` is the game the answer is *about*; the evidence is every game the
    # comparison ran over. The two used to be the same, which under-cited: "the
    # largest margin of defeat" was decided by reading all 18 defeats and only
    # one was listed. The README asks for every contributing row, so all of them
    # are cited while the score and opponent still come from the winner.
    focus = None
    if plan["source"] == "game_details" and agg["op"] in ("max", "min"):
        field_of = lambda g: float(g.get(agg["field"]) or 0)
        target = nth_value([field_of(g) for g in games], agg.get("rank", 1),
                           agg["op"] == "max")
        if target is None:
            result.empty = True
            return result
        # Lowest game_id among ties, so a rerun picks the same game.
        focus = min((g for g in games if field_of(g) == target),
                    key=lambda g: int(g["game_id"]))
        result.values["agg"] = target

    # "Which team did they beat most often" ranks opponents rather than players,
    # so the grouping key is the other team in each game.
    if agg["group_by"] == "opponent_team_id" and agg["op"] in ("argmax", "argmin"):
        chosen = _argmax_by_opponent(result, games, agg, subject_team, teams)
        if not chosen:
            result.empty = True
            return result
        for game in chosen:
            result.touch_game(game)
        result.values["n_games"] = len(chosen)
        result.values["n_rows"] = len(chosen)
        result.values["subject_team"] = subject_team
        return result

    contributing = games
    if plan["source"] == "player_box_scores":
        box_rows = select_box_rows(cx, [g["game_id"] for g in games], plan["row_filters"])
        if not box_rows:
            result.empty = True
            return result

        if agg["op"] in ("argmax", "argmin"):
            cited = _apply_argmax(result, box_rows, agg, players)
            if not cited:
                result.empty = True
                return result
        else:
            cited = box_rows
            values = [box_field(r, agg["field"]) for r in box_rows]
            if agg["op"] == "count":
                result.values["agg"] = len(box_rows)
            elif agg["op"] == "count_games":
                result.values["agg"] = len({int(r["game_id"]) for r in box_rows})
            elif agg["op"] == "avg":
                # Per game, not per row: the phrase people mean by "average" is
                # points per game, and a player has one line per game.
                n_games = len({int(r["game_id"]) for r in box_rows}) or 1
                result.values["agg"] = sum(values) / n_games
            elif agg["op"] in ("max", "min"):
                target = nth_value(values, agg.get("rank", 1), agg["op"] == "max")
                if target is None:
                    result.empty = True
                    return result
                result.values["agg"] = target
                # A per-row extreme cites the row that produced the number. The
                # strict reading is applied to comparisons over *games* (see the
                # game-level max above and the recap scan in rag.py), where the
                # candidate set is the answer's scope; applying it here instead
                # cited a player's whole season -- 136 rows -- to report his
                # single best night.
                cited = [r for r in box_rows if box_field(r, agg["field"]) == target]
            else:
                result.values["agg"] = sum(values)

        for row in cited:
            result.touch_box(row)
        cited_game_ids = {int(r["game_id"]) for r in cited}
        contributing = [g for g in games if int(g["game_id"]) in cited_game_ids]
        result.values["n_rows"] = len(cited)
        result.values["n_games"] = len(cited_game_ids)

        # Which team(s) the cited rows belong to. 69 of 569 players appear for
        # more than one team in this season, so "how many teams did X play for"
        # is a real question, and answering it with a game count -- which is
        # what happened before there was a primitive for it -- is not close.
        cited_teams = sorted({int(r["team_id"]) for r in cited})
        result.values["n_teams"] = len(cited_teams)
        result.values["team_name"] = ", ".join(
            teams.get(t, {}).get("name", "") for t in cited_teams)
        # Distinct people, not rows: "how many different players scored 30+"
        # counts players, while the row count counts performances.
        result.values["n_players"] = len({int(r["person_id"]) for r in cited})
    elif plan["source"] == "injury_notes":
        # Counting absences is a question about notes, not games: "how many
        # games did X miss" is the number of games they were listed out for.
        # Only a question counting *missed* games should restrict to "out".
        # "How was he listed" is asking which status he carried, and filtering
        # to "out" made every questionable and probable note invisible -- so a
        # player who was listed questionable and played came back unanswerable.
        wants_status = any(f in plan["outputs"] for f in ("status", "reason"))
        notes = count_notes(cx, [g["game_id"] for g in games],
                            plan["row_filters"].get("person_id"),
                            None if wants_status else "out")
        if not notes:
            result.empty = True
            return result

        # "Who missed the most games" ranks players by absences, the same shape
        # as any other argmax but over notes rather than box scores.
        if agg["op"] in ("argmax", "argmin") and agg["group_by"] == "person_id":
            tally: dict[int, int] = {}
            for note in notes:
                pid = int(note["player_id"])
                tally[pid] = tally.get(pid, 0) + 1
            best = nth_value(list(tally.values()), agg.get("rank", 1),
                             agg["op"] == "argmax")
            if best is None:
                result.empty = True
                return result
            tied = sorted(p for p, c in tally.items() if c == best)
            result.values["agg"] = best
            result.values["group_name"] = tied_names(tied, players)
            notes = [n for n in notes if int(n["player_id"]) in set(tied)]

        for note in notes:
            result.notes.add(int(note["note_id"]))
        note_games = {int(n["game_id"]) for n in notes}
        contributing = [g for g in games if int(g["game_id"]) in note_games]
        result.values["n_rows"] = len(notes)
        result.values["n_games"] = len(note_games)
        result.values.setdefault("agg", len(note_games))
        result.values["n_players"] = len({int(n["player_id"]) for n in notes})
    else:
        result.values.setdefault("n_rows", len(games))
        result.values["n_games"] = len(games)
        if agg["op"] in ("count", "count_games"):
            result.values["agg"] = len(games)
        elif agg["op"] == "avg":
            # A per-game average over games, e.g. points allowed per game.
            values = [float(g.get(agg["field"]) or 0) for g in games]
            result.values["agg"] = sum(values) / len(values)
        elif agg["op"] == "sum":
            result.values["agg"] = sum(float(g.get(agg["field"]) or 0) for g in games)

    for game in contributing:
        result.touch_game(game)

    # A win-loss record needs both halves at once, so they are always available
    # rather than requiring the question to be asked twice with opposite filters.
    if subject_team is not None:
        wins = sum(1 for g in contributing
                   if g.get("winning_team_id") is not None
                   and int(g["winning_team_id"]) == int(subject_team))
        result.values["n_wins"] = wins
        result.values["n_losses"] = len(contributing) - wins

    # Facts about "the game" come from the focus game when a comparison picked
    # one, and otherwise only when the plan narrowed to exactly one. Leaving
    # them unset makes the caller abstain instead of reporting an arbitrary
    # game's score.
    if focus is not None:
        apply_single_game_values(result, focus, subject_team, teams)
    elif len(contributing) == 1:
        apply_single_game_values(result, contributing[0], subject_team, teams)

    result.values["subject_team"] = subject_team
    return result


def apply_single_game_values(result: ExecResult, game: dict, subject_team: int | None,
                             teams: dict[int, dict]) -> None:
    """Fill in the facts that only make sense for one specific game.

    Split out so the text path can call it again after narrowing a season-wide
    scan down to the single game that won it -- at which point the score and the
    opponent become answerable from the table, having not been before.

    :param result: ExecResult to write into
    :param game: the chosen game row
    :param subject_team: team the question is about, for naming the opponent
    :param teams: team_id -> team dict
    """
    result.values["score"] = format_score(game)
    result.values["game_date"] = str(game["game_date"])
    result.values["margin"] = int(game["margin"])
    winner_id = game.get("winning_team_id")
    if winner_id is not None:
        result.values["winner_name"] = teams.get(int(winner_id), {}).get("name", "")
    opponent_id = opponent_of(game, subject_team)
    if opponent_id is not None:
        result.values["opponent_name"] = teams.get(opponent_id, {}).get("name", "")
