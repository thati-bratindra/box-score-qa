"""A small relational core: columns, predicates, and expressions.

This exists because the plan DSL was growing a named key per question shape.
`min_points`, `max_points`, `min_rebounds`, `min_assists` and `min_margin` are
all one idea -- compare a column to a value -- and each arrived with its own
branch in the query builder. `n_games`, `n_players`, `n_teams` and `n_rows` are
all `count distinct` over different columns. Twenty-odd features, four ideas.

So the query builder takes a list of `{col, op, val}` predicates and a list of
`{as, fn, col}` expressions, and a registry says what each column *is* and what
it needs joined. Adding "most blocks by a center in road games" becomes a
vocabulary entry rather than new code here, because blocks, position and venue
are already columns.

Three things genuinely resist this and stay as named relations in SPECIALS:
outcome, venue and conference are comparisons against the *subject team* rather
than a constant, and absence is an anti-join. They are declared in one place
instead of being scattered through the builder.
"""
from sqlalchemy import text
from sqlalchemy.engine import Connection

# ingest.py stores game_timestamp as text, so every date comparison casts. The
# cast is to a naive timestamp on purpose: no game tips off late enough to cross
# midnight, and converting to UTC first would shift evening games a day forward.
GAME_DATE = "g.game_timestamp::timestamp::date"

# column name -> (SQL expression, join key needed to reach it)
BOX_COLUMNS = {
    "game_id": ("b.game_id", None),
    "person_id": ("b.person_id", None),
    "team_id": ("b.team_id", None),
    "starter": ("b.starter", None),
    "seconds": ("b.seconds", None),
    "points": ("b.points", None),
    "assists": ("b.assists", None),
    "steals": ("b.steals", None),
    "blocks": ("b.blocks", None),
    "turnovers": ("b.turnovers", None),
    "offensive_reb": ("b.offensive_reb", None),
    "defensive_reb": ("b.defensive_reb", None),
    "rebounds": ("(b.offensive_reb + b.defensive_reb)", None),
    "fg2_made": ("b.fg2_made", None),
    "fg2_attempted": ("b.fg2_attempted", None),
    "fg3_made": ("b.fg3_made", None),
    "fg3_attempted": ("b.fg3_attempted", None),
    "ft_made": ("b.ft_made", None),
    "ft_attempted": ("b.ft_attempted", None),
    "game_date": (GAME_DATE, "game"),
    "winning_team_id": ("g.winning_team_id", "game"),
    "home_team_id": ("g.home_team_id", "game"),
    "away_team_id": ("g.away_team_id", "game"),
    "position": ("p.position", "player"),
    "last_name": ("p.last_name", "player"),
}

GAME_COLUMNS = {
    "game_id": ("g.game_id", None),
    "season": ("g.season", None),
    "home_team_id": ("g.home_team_id", None),
    "away_team_id": ("g.away_team_id", None),
    "home_points": ("g.home_points", None),
    "away_points": ("g.away_points", None),
    "winning_team_id": ("g.winning_team_id", None),
    "game_date": (GAME_DATE, None),
    "margin": ("ABS(g.home_points - g.away_points)", None),
    "total_points": ("(g.home_points + g.away_points)", None),
}

SOURCES = {
    "player_box_scores": {
        "base": "player_box_scores b",
        "columns": BOX_COLUMNS,
        "joins": {
            "game": "JOIN game_details g ON g.game_id = b.game_id",
            "player": "JOIN players p ON p.player_id = b.person_id",
        },
        "order": "b.game_id, b.person_id",
        "select": "b.*",
    },
    "game_details": {
        "base": "game_details g",
        "columns": GAME_COLUMNS,
        "joins": {},
        "order": f"g.game_timestamp::timestamp, g.game_id",
        "select": (
            "g.game_id, g.home_team_id, g.away_team_id, g.home_points, "
            "g.away_points, g.winning_team_id, "
            f"{GAME_DATE} AS game_date, "
            "(g.home_points + g.away_points) AS total_points, "
            "ABS(g.home_points - g.away_points) AS margin"
        ),
    },
}

# Comparison operators, mapped to SQL. Kept to a whitelist so a malformed plan
# can never interpolate an operator of its own choosing.
OPERATORS = {
    "=": "=", "==": "=", "!=": "<>", "<>": "<>",
    ">": ">", ">=": ">=", "<": "<", "<=": "<=",
}

# Relations that compare against the subject team rather than a constant, plus
# the anti-join for absence. Each takes the value as :val and, where relevant,
# the subject team as :team.
SPECIALS = {
    # A team played in this game, on either side.
    "played_in": "({t}.home_team_id = :{p} OR {t}.away_team_id = :{p})",
    "won": "{t}.winning_team_id = :{p}",
    "lost": "{t}.winning_team_id <> :{p}",
    "hosted": "{t}.home_team_id = :{p}",
    "visited": "{t}.away_team_id = :{p}",
}


def _resolve(source: str, col: str) -> tuple[str, str | None]:
    """Look a column up in the registry.

    :param source: source table name
    :param col: column name as written in a predicate
    :return: (SQL expression, required join key)
    :raises KeyError: if the column is not registered for this source
    """
    return SOURCES[source]["columns"][col]


def build_where(source: str, predicates: list[dict]) -> tuple[list[str], dict, set]:
    """Turn a predicate list into SQL fragments.

    :param source: source table name
    :param predicates: list of {col, op, val}; `op` may also be "in", "not_in",
        "between", or the name of a SPECIALS relation
    :return: (where fragments, bind params, join keys required)
    :raises KeyError: on an unregistered column
    :raises ValueError: on an unrecognized operator
    """
    where: list[str] = []
    params: dict = {}
    joins: set = set()
    alias = "b" if source == "player_box_scores" else "g"

    for i, pred in enumerate(predicates):
        op = pred.get("op", "=")
        value = pred.get("val")
        key = f"p{i}"

        if op in SPECIALS:
            # A team-relative relation. On box scores these live on the joined
            # game row, so the game join comes along with them.
            table = "g"
            if source == "player_box_scores":
                joins.add("game")
            where.append(SPECIALS[op].format(t=table, p=key))
            params[key] = int(value)
            continue

        if op == "absent":
            where.append("NOT EXISTS (SELECT 1 FROM player_box_scores x "
                         f"WHERE x.game_id = {alias if source == 'game_details' else 'g'}.game_id "
                         f"AND x.person_id = :{key})")
            if source == "player_box_scores":
                joins.add("game")
            params[key] = int(value)
            continue

        if op == "opponent_conference":
            table = "g"
            if source == "player_box_scores":
                joins.add("game")
            where.append(
                "EXISTS (SELECT 1 FROM teams ct WHERE ct.team_id = CASE "
                f"WHEN {table}.home_team_id = :{key}_t THEN {table}.away_team_id "
                f"ELSE {table}.home_team_id END AND ct.conference = :{key}_c)")
            params[f"{key}_t"] = int(value[0])
            params[f"{key}_c"] = str(value[1])
            continue

        expr, join = _resolve(source, pred["col"])
        if join:
            joins.add(join)

        if op == "in" or op == "not_in":
            negate = "NOT " if op == "not_in" else ""
            where.append(f"{expr} {negate}= ANY(:{key})")
            params[key] = list(value)
        elif op == "between":
            where.append(f"{expr} BETWEEN :{key}_lo AND :{key}_hi")
            params[f"{key}_lo"], params[f"{key}_hi"] = value[0], value[1]
        elif op == "like":
            where.append(f"{expr} LIKE :{key}")
            params[key] = value
        elif op in OPERATORS:
            where.append(f"{expr} {OPERATORS[op]} :{key}")
            params[key] = value
        else:
            raise ValueError(f"Unknown predicate operator: {op}")

    return where, params, joins


def select_rows(cx: Connection, source: str, predicates: list[dict]) -> list[dict]:
    """Run a predicate list against a source and return the matching rows.

    Ordering is fixed per source and always ends in a primary-key tie-break, so
    two runs over the same data return rows in the same order -- which the
    reproducibility check depends on.

    :param cx: open SQLAlchemy connection
    :param source: "player_box_scores" or "game_details"
    :param predicates: list of {col, op, val}
    :return: matching rows as dicts
    """
    spec = SOURCES[source]
    where, params, joins = build_where(source, predicates)
    join_sql = " ".join(spec["joins"][j] for j in sorted(joins) if j in spec["joins"])
    sql = (f"SELECT {spec['select']} FROM {spec['base']} {join_sql} "
           + (("WHERE " + " AND ".join(where) + " ") if where else "")
           + f"ORDER BY {spec['order']}")
    return [dict(r) for r in cx.execute(text(sql), params).mappings().all()]


def evaluate(rows: list[dict], fn: str, col: str | None,
             value_of=None) -> float | int | None:
    """Apply an aggregate function to already-fetched rows.

    Aggregating in Python rather than SQL keeps the rows that produced the
    number in hand, which is what the evidence list is built from. The row set
    here is at most a season of one team's games, so there is nothing to gain by
    pushing it down.

    :param rows: rows to aggregate
    :param fn: sum, avg, count, count_distinct, min, max
    :param col: column to aggregate, or None for count
    :param value_of: optional accessor, for derived columns the registry does
        not compute in SQL
    :return: the aggregate, or None when there is nothing to aggregate
    """
    if fn == "count":
        return len(rows)
    if not rows:
        return None
    read = value_of or (lambda r: float(r.get(col) or 0))
    if fn == "count_distinct":
        return len({r.get(col) for r in rows})
    values = [read(r) for r in rows]
    if fn == "sum":
        return sum(values)
    if fn == "avg":
        return sum(values) / len(values)
    if fn == "min":
        return min(values)
    if fn == "max":
        return max(values)
    raise ValueError(f"Unknown aggregate function: {fn}")
