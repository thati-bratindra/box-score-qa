"""Turn a question into a declarative query plan.

The LLM's only job here is to fill in a small fixed form: which table to read,
how to narrow it, how to combine the rows, and which result goes in which field
of the question's `return` block. It never writes SQL, never writes code, and
never produces a value that reaches the answer -- execute.py computes every
number from the tables.

That split is what makes a 2B model workable. Choosing "sum points, group by
player, take the max" is a classification problem with a handful of options;
computing the sum correctly over 40 rows is not something a 2B model should be
trusted with, and it does not have to be.

Two things keep a weak model from doing damage:

* **Pre-fill.** Entities are resolved deterministically first (see resolve.py)
  and a skeleton plan is derived from them. The model refines that skeleton; it
  does not start from nothing.
* **Closed-world ids.** validate_plan rejects any person_id or team_id that did
  not come out of resolution, so the model cannot invent an id that happens to
  exist in the database and silently answer about the wrong player.

A plan that fails validation twice is not repaired or guessed at -- the caller
abstains.
"""
import json
import re

from backend.resolve import (MONTHS, PROPER_NOUN_STOPWORDS, SUBJECT_STOPWORDS,
                             whole_word_match)

# Question wording that pins down part of the plan without any model judgement.
# These are general English patterns -- a threshold, an outcome, an ordinal --
# not question-specific rules, so they carry over to new questions phrased
# differently about different teams and players.
#
# They exist because a 2B model is unreliable at exactly this: an early version
# let the model set the filters and it dropped "30 or more points" from one
# question and "the Outriders" from another, producing confident answers to
# questions nobody asked. A regex that finds "30 or more points" is simply more
# trustworthy than qwen3.5:2b, so where the two disagree, the regex wins.
THRESHOLD_PATTERNS = [
    r"(\d+)\s*\+?\s*or\s+more\s+points",
    r"at\s+least\s+(\d+)\s+points",
    r"(\d+)\s+or\s+more\s+point",
    r"(\d+)\+\s+points",
]
# "More than 30" excludes 30; "30 or more" includes it. Points are integers, so
# the strict form is exactly the inclusive filter one higher. Without these the
# phrase matched nothing at all and the threshold silently vanished, turning
# "most games with more than 30 points" into "most games played".
THRESHOLD_EXCLUSIVE_PATTERNS = [
    r"more\s+than\s+(\d+)\s+points",
    r"over\s+(\d+)\s+points",
    r"above\s+(\d+)\s+points",
]
MARGIN_PATTERNS = [
    r"by\s+(\d+)\s+or\s+more\s+points",
    r"by\s+more\s+than\s+(\d+)\s+points",
    r"by\s+at\s+least\s+(\d+)",
    r"by\s+(\d+)\+",
]
# The same idea pointing the other way: a ceiling on the margin, which is how
# people ask about close games. Without it the condition was recognized by
# nobody and dropped, so "how many games were decided by 3 points or fewer"
# counted all 841 games in the season.
MAX_MARGIN_INCLUSIVE = [
    # The noun sits between the number and the comparison in the way people
    # actually write this -- "by 3 points or fewer" -- so it has to be optional
    # rather than absent, which is what let the whole condition fall through.
    r"by\s+(\d+)\s+(?:points?\s+)?or\s+(?:fewer|less)\b",
    r"by\s+no\s+more\s+than\s+(\d+)",
    r"by\s+at\s+most\s+(\d+)",
    r"within\s+(\d+)\s+points?",
]
MAX_MARGIN_EXCLUSIVE = [
    r"by\s+(?:fewer|less)\s+than\s+(\d+)",
    r"by\s+under\s+(\d+)",
]
# "who won?" is a question, not a statement that the subject won, so the bare
# past tense "won" stays out of the win vocabulary. "won by", though, is always
# a claim about the result -- leaving it out meant "games the Outriders won by 10
# or more" aggregated over their losses too.
# The base form belongs here for the same reason "win" does. Without it, "how
# many road games did the Outriders lose" set no outcome at all and counted all 40
# of their road games instead of the 10 they lost -- a real number, filtered by
# venue exactly as asked, answering half the question.
LOSS_WORDS = (r"\blost\b|\blose\b|\bloses\b|\blosses\b|\bloss\b|\blosing\b"
              r"|\bdefeat\b|\bdefeats\b|\bdefeated\b|\bbeaten\b")
WIN_WORDS = (r"\bwins\b|\bwin\b|\bvictory\b|\bvictories\b|\bwinning\b"
             r"|\bwon\s+by\b|\bwon\s+against\b|\bwon\s+over\b")
FIRST_WORDS = r"season opener|first game|opening game|season-opening|opening night|earliest game"
LAST_WORDS = r"final game|last game|season finale|most recent game|closing game"

# "Their 10th game" is a position in the schedule, not a ranking. Distinct from
# the ordinals in ORDINAL_WORDS, which only count when a superlative follows.
NTH_GAME_PATTERN = (r"\b(\d+)(?:st|nd|rd|th)\s+game\b"
                    r"|\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+game\b")
NTH_WORDS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

# A "closest game" is the smallest margin and a "blowout" the largest. Without
# these, "the Outriders' closest game" came back as their 49-point rout.
# "Largest margin of victory" and "worst loss by point differential" are both
# single-game extremes of the margin. They need saying explicitly: "worst"
# otherwise reads as a minimum, which returned the Outriders' *narrowest* defeat,
# and "against which opponent" otherwise reads as a grouping over opponents.
MARGIN_EXTREME_WORDS = (r"\bmargin of (?:victory|defeat)\b|\bworst (?:loss|defeat)\b"
                        r"|\b(?:largest|biggest|greatest|widest)\s+margin\b"
                        r"|\bpoint differential\b")
CLOSEST_WORDS = r"\bclosest\b|\bnarrowest\b|\btightest\b|\bnail[-\s]?biter\b|\bcloser\b"
BLOWOUT_WORDS = r"\bblowout\b|\bmost lopsided\b|\bbiggest rout\b|\bmost one[-\s]sided\b"
STARTER_WORDS = r"\bstarter\b|\bstarters\b|\bstarting\b"
BENCH_WORDS = r"\bbench\b|\boff the bench\b|\breserves?\b|\bsecond unit\b|\bsubstitutes?\b|\bbackup\b"
# "Start" as the verb, which the nouns above cannot see. "How many games did
# Hollis Whitcombe start" set no starter filter and answered 71 -- every game he
# played -- where he started 9. Only the verb applied to a *game* counts: "the
# start of February" and "to start the season" are about the calendar, not the
# lineup, so each form needs "game" nearby and none may run on into "of",
# "to", "off" or "the season".
_NOT_CALENDAR = r"(?!\s+(?:of|to|off|the\s+(?:season|year|month|week))\b)"
# The words between "did" and "start" have to be its subject -- "Kamari
# Whitcombe", "he". If one of them is another verb or a preposition, "start" is
# not what the subject did: "did they win at the start" is about the calendar.
_SUBJECT_WORD = (r"(?:(?!(?:win|won|lose|lost|score|scored|play|played|at|after|since"
                 r"|before|following|despite|until|with|from)\b)[\w'’.-]+\s+)")
STARTED_VERB_WORDS = (
    # "games did Hollis Whitcombe start", "games has he started"
    rf"\bgames?\s+(?:did|does|has|have|had)\s+{_SUBJECT_WORD}{{1,4}}?"
    rf"(?<!the )(?<!a )start(?:ed)?\b{_NOT_CALENDAR}"
    # "games Whitcombe started", "games in which he started"
    rf"|\bgames?\s+(?:that\s+|in\s+which\s+)?{_SUBJECT_WORD}{{1,3}}?"
    rf"(?<!the )(?<!a )started\b{_NOT_CALENDAR}"
    # "started the most games", "started every game", "started 50 games"
    r"|\bstart(?:ed|s)?\s+(?:in\s+)?(?:the\s+most|the\s+fewest|more|fewer|every|all|each|\d+)"
    r"\s+(?:of\s+(?:the|their|his|her)\s+)?(?:\w+\s+)?games?\b"
)
# Checked before the verb: "games he did not start" are bench games, and the
# verb pattern alone would read them as starts.
NOT_STARTED_WORDS = (r"\b(?:not|never|didn['’]?t|doesn['’]?t|hasn['’]?t|haven['’]?t)"
                     r"\s+start(?:ed)?\b")

# Roster positions, from players.position ("G", "F", "C", and hyphenated
# combinations like "G-F" -- a guard-forward counts as both).
POSITION_WORDS = [
    (r"\bguards?\b|\bpoint guard\b|\bshooting guard\b|\bbackcourt\b", "G"),
    (r"\bforwards?\b|\bsmall forward\b|\bpower forward\b|\bwing\b", "F"),
    (r"\bcent(?:er|re)s?\b|\bbig man\b|\bfrontcourt\b", "C"),
]

# A streak is a run of consecutive results, which needs the games in order
# rather than a group-and-aggregate.
STREAK_WORDS = r"\bstreaks?\b|\bconsecutive\b|\bin a row\b|\bstraight\b|\brun of\b"

# Which conference the opponent belongs to. Teams carry a conference column, and
# only games involving a Western team are in the data at all, so "against
# Eastern Conference teams" is a real and answerable split.
EAST_WORDS = r"\beastern conference\b|\bthe east\b|\beast(?:ern)? teams?\b"
WEST_WORDS = r"\bwestern conference\b|\bthe west\b|\bwest(?:ern)? teams?\b"

# "The fewest" is the same question as "the most" pointed the other way, and
# getting the direction wrong returns the opposite player with full confidence.
# "At least 15" is a threshold, not a direction. Matching the bare word "least"
# there flipped "which starter piled up the most points" into an argmin and
# returned the lowest scorer -- 22 points where the answer was 142.
FEWEST_WORDS = (r"\bfewest\b|\blowest\b|(?<!at )\bleast\b|\bworst\b|\bminimum\b"
                r"|\bsmallest\b|\bpoorest\b")
# Vocabulary that asks for a ranking. Its absence is what separates "which
# bench player scored the most" from "which bench players scored more than 10":
# the first names one person, the second names a set, and answering the second
# with the top one of them is answering a question nobody asked.
RANKING_WORDS = (r"\bmost\b|\bfewest\b|\bhighest\b|\blowest\b|\bbest\b|\bworst\b"
                 r"|(?<!at )\bleast\b|\btop\b|\bleading\b|\bleader\b|\bled\b"
                 r"|\blargest\b|\bbiggest\b|\bsmallest\b|\bgreatest\b|\bmaximum\b"
                 r"|\bminimum\b|\bmore than any\b|\bmost often\b|\bmost frequently\b"
                 r"|\bsecond[-\s]?most\b|\bthird[-\s]?most\b|\bpaced\b|\btopped\b")

# "In a game" asks for a single-game extreme, not a season total: "the fewest
# points in a game" is somebody's quiet night, while "the fewest points" is a
# season-long total. The two have different answers.
SINGLE_GAME_WORDS = r"\bin a (?:single )?game\b|\bin one game\b|\bsingle[-\s]game\b|\bin any game\b"

DOUBLE_DOUBLE_WORDS = r"\bdouble[-\s]doubles?\b"
TRIPLE_DOUBLE_WORDS = r"\btriple[-\s]doubles?\b"

# "Which team ..." groups by opponent rather than by player.
TEAM_SUBJECT_WORDS = r"\bwhich team\b|\bwhat team\b|\bwhich opponent\b|\bwhich club\b"

# "Who beat them" is not by itself a ranking. Q11 asks for the largest margin of
# defeat and mentions "who beat them" only to name that one opponent -- grouping
# by opponent there would total every defeat per team and answer something else.
# So this phrasing only ranks when the question also asks how *often*.
WHO_BEAT_WORDS = r"\bwho\s+(?:beat|defeated|won\s+against|knocked off)\b"
FREQUENCY_WORDS = (r"\bmost often\b|\bmost frequently\b|\bhow many times\b"
                   r"|\bmost times\b|\bmost\s+\w+\s+times\b")

# ---------------------------------------------------------------------------
# What this pipeline can answer -- a whitelist, deliberately, not a blacklist.
#
# The set of things these six tables do not describe is unbounded: age, height,
# salary, coaches, attendance, draft position, what a player is like. Listing
# them can never be finished, and whatever is missing from the list gets
# answered anyway -- "how old is Tavian Belcourt" came back as "64 points",
# because an unrecognized question fell through to a default that assumed every
# question was about scoring.
#
# The set of operations implemented here, by contrast, is finite and already
# written down elsewhere in this file. So the test is inverted: a question must
# name something the pipeline can actually produce or rank by. Naming nothing
# is not a question this code has an operation for, whatever it is about.
#
# A word appearing here only means "do not refuse outright" -- every filter,
# audit and grounding check downstream still applies. So a loose match costs
# nothing beyond today's behaviour, while a missing one costs a refusal, which
# is the safe direction to err in.
SUPPORTED_CONCEPT_WORDS = (
    # Statistics the box scores hold and this code can aggregate.
    r"\bpoints?\b|\bscor\w*|\bbuckets?\b|\brebound\w*|\bboards\b|\bglass\b"
    r"|\bassist\w*|\bdimes\b|\bpass(?:er|ing)\b|\bsteal\w*|\bblock\w*"
    r"|\bturnover\w*|\bgiveaway\w*|\bminutes?\b|\bplaying time\b"
    r"|\bthree[-\s]?point\w*|\bthrees\b|\btriples\b|\bfree[-\s]?throw\w*"
    r"|\bpercentage\b|\bpct\b|\bshooting\b|\bfg%|\b3p%|\bft%"
    r"|\bdouble[-\s]doubles?\b|\btriple[-\s]doubles?\b"
    # Facts about a game itself.
    r"|\bscore\b|\bfinal\b|\bwinner\b|\bwon\b|\bwin\b|\bwins\b|\blost\b|\bloss\b"
    r"|\blosses\b|\blosing\b|\bbeat\b|\bdefeat\w*|\bopponent\b|\bmargin\b"
    r"|\bdifferential\b|\bblowout\b|\bclosest\b|\bnarrowest\b|\btightest\b"
    r"|\brecord\b|\bstreak\b|\bconsecutive\b|\bin a row\b|\bstraight\b"
    # Facts that live only in the written recaps.
    r"|\bdeficit\b|\bhole\b|\bcomeback\b|\bcame back\b|\brallied\b|\bovercame\b"
    r"|\berased\b|\bclimbed out\b|\bgo-ahead\b|\bgame[-\s]winner\b|\bclutch\b"
    # Availability, from the injury notes.
    r"|\bout\b|\bmiss\w*|\binjur\w*|\b(?:un)?available\b|\blisted\b|\bstatus\b"
    r"|\bquestionable\b|\bdoubtful\b|\bprobable\b|\bsidelined\b|\bsat out\b"
    r"|\brest\b|\binactive\b"
    # Aggregate vocabulary, which only counts alongside one of the above.
    r"|\bhow many\b|\bhow much\b|\bmost\b|\bfewest\b|\bhighest\b|\blowest\b"
    r"|\btotal\b|\baverage[ds]?\b|\bavg\b|\bled\b|\bleading\b|\bleader\b"
    r"|\blargest\b|\bbiggest\b|\bworst\b"
)

# Count nouns are too ordinary to admit on their own -- "player" appears in
# "who is the tallest player", which this pipeline cannot answer. They count as
# a supported concept only where the question is plainly counting them.
SUPPORTED_COUNT_PATTERN = (
    r"\b(?:how many|number of|most|fewest|different|distinct)\s+"
    r"(?:\w+\s+){0,2}?(?:games?|teams?|players?|wins?|losses|times|nights?)\b"
)


def supported_concept(question_text: str) -> bool:
    """Does the question name something this pipeline has an operation for?

    :param question_text: the raw question
    :return: True when the question is worth planning at all
    """
    return bool(re.search(SUPPORTED_CONCEPT_WORDS, question_text, re.IGNORECASE)
                or re.search(SUPPORTED_COUNT_PATTERN, question_text, re.IGNORECASE))


# The only blacklist still needed. These questions ask for something the
# pipeline *does* support -- points -- and hide the unsupported part in a
# condition, so no whitelist on the answer can catch them: "points in the third
# quarter" was answered with the full-game total, a real number for a question
# nobody asked. Everything the old blacklist covered besides this (height, age,
# draft year, salary, coaches, attendance) now fails the whitelist on its own.
UNSUPPORTED_CONCEPTS = [
    (r"\bquarter\b|\bquarters\b|\bfirst half\b|\bsecond half\b|\bhalftime\b"
     r"|\bhalf-time\b|\bperiod\b|\bovertime\b|\bfourth\b\s+\bquarter\b",
     "the box scores hold whole-game totals only, with no period splits"),
]


def unsupported_concept(question_text: str) -> str | None:
    """Detect a question asking for something the data cannot support.

    Returning the reason rather than a bare flag keeps the abstention
    explainable, which matters more here than anywhere else: the answer is not
    "I could not work it out" but "the tables do not contain this".

    :param question_text: the raw question
    :return: a short reason, or None when the question is answerable in principle
    """
    for pattern, reason in UNSUPPORTED_CONCEPTS:
        if re.search(pattern, question_text, re.IGNORECASE):
            return reason
    return None

# An average is a different question from a total, and answering one with the
# other is silent and confident. "Per game" is the giveaway that survives
# rewording; "most games" does not match it, since that is "games" without "per".
#
# The "<stat> a game" form is spelled out stat by stat on purpose. A bare
# "a game" matched "overcame to win a game this season" and turned a question
# about a comeback into an average, which is the kind of false positive that
# costs a whole answer.
AVERAGE_WORDS = (r"\baverage[ds]?\b|\bavg\b|\bper\s+game\b|\bper\s+contest\b"
                 r"|\b(?:points|rebounds|assists|boards|steals|blocks)\s+a\s+game\b")

# "Second-highest scorer" asks for rank 2, not rank 1. The superlative after the
# ordinal is required so that "second half" and "second quarter" -- which appear
# all over the recaps -- are not read as a ranking.
ORDINAL_WORDS = {
    "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
}
RANK_SUPERLATIVES = r"highest|most|largest|biggest|greatest|best|leading|top|worst|fewest|lowest"

# The two had drifted out of symmetry: HOME_WORDS recognizes a bare "at home",
# AWAY_WORDS only recognized "away" when a noun followed it. So "how many games
# did the Trappers play away against the Mammoths" matched nothing, the venue
# filter was never built, and the question silently became "against the
# Mammoths" -- answering 2 where the away answer is 1, while its "at home"
# twin answered correctly. Adverbial "away" is now covered in both voices.
HOME_WORDS = (r"\bat home\b|\bhome (?:game|record|court|floor)|\bhosting\b"
              r"|\bas hosts?\b|\bhome\s+(?:at|against|versus|vs\.?)\b"
              r"|\bplay(?:s|ed|ing)?\s+at\s+home\b")
AWAY_WORDS = (r"\bon the road\b|\broad (?:game|record|trip)|\baway (?:game|record)\b"
              r"|\bas visitors?\b|\baway\s+(?:at|against|versus|vs\.?|from home)\b"
              r"|\bplay(?:s|ed|ing)?\s+away\b|\baway\s+(?:win|wins|loss|losses)\b")

# "Games the Outriders played without X" asks about games X is absent from, which
# is the opposite of every other player filter: it constrains the game set by a
# player's *non*-appearance rather than selecting that player's rows.
WITHOUT_WORDS = (r"\bwithout\b|\bmissed\b|\bmissing\b|\bsat out\b|\bdid not play\b"
                 r"|\bdidn't play\b|\bin the absence of\b|\babsent\b")

# Which stat a question is about. Matched against the question's own wording so
# that a new question asking about rebounds does not get answered with
# points, which is what happens when the field is left to the model. Ordered:
# percentage phrasings are checked before the raw makes they contain.
BOX_FIELD_WORDS = [
    (r"three[-\s]?point(?:er)?s?\s*(?:%|percentage|pct|shooting)|\b3p%|\bthree[-\s]?ball\b", "fg3_pct"),
    (r"free[-\s]?throw\s*(?:%|percentage|pct|shooting)|\bft%", "ft_pct"),
    (r"field[-\s]?goal\s*(?:%|percentage|pct|shooting)|\bfg%|shooting percentage", "fg_pct"),
    # People name a stat by the role as often as by the column: a "leading
    # passer" is asking about assists, and answering with points is answering a
    # different question.
    (r"\brebound|\bboards\b|\bglass\b|\brebounder\b|\bleading rebounder\b", "rebounds"),
    (r"\bassist|\bdimes\b|\bplaymak|\bpasser\b|\bpassing\b|\bfacilitator\b", "assists"),
    (r"\bsteal", "steals"),
    (r"\bblock", "blocks"),
    (r"\bturnover|\bgiveaway", "turnovers"),
    (r"three[-\s]?point(?:er)?s?\b|\b3[-\s]?point|\bthrees\b|\btriples\b", "fg3_made"),
    (r"\bfree[-\s]?throws?\b", "ft_made"),
    (r"\bminutes\b|\bplaying time\b", "seconds"),
    (r"\bpoints?\b|\bscor|\bbuckets\b", "points"),
]
GAME_FIELD_WORDS = [
    (r"\ballow(?:ed)?\b|\bgave up\b|\bconce(?:de|ded)\b|\bsurrendered\b"
     r"|\bopponents? scored\b|\bdefensive\b", "points_allowed"),
    (r"\bmargin\b|\bblowout\b|\bbeat them by\b|\bdefeat(?:ed)? them by\b", "margin"),
    (r"\bcombined (?:points|score)\b|\btotal points\b", "total_points"),
    (r"\bscored\b|\bpoints\b", "points_for"),
]

# Two filter scopes, because ordinals and row filters operate at different
# levels. "The Outriders' season opener" is a property of the team's game list; the
# player's line is then looked up *inside* that game. Collapsing both into one
# filter set would make "first" mean "this player's first game", which for a
# bench player is a different game entirely -- and would return a confident
# wrong number instead of an abstention.
GAME_FILTER_KEYS = {
    "team_id", "opponent_team_id", "game_id", "date_from", "date_to",
    "outcome", "min_margin", "max_margin", "venue", "exclude_person_id",
    "opponent_conference",
}
ROW_FILTER_KEYS = {"person_id", "person_ids", "team_id", "exclude_team_id",
                   "starter", "min_points", "max_points", "min_rebounds",
                   "min_assists", "double_double", "triple_double", "position"}
POSITIONS = {"G", "F", "C"}
VENUES = {"home", "away"}

# Which row filters each source actually applies. injury_notes is filtered by
# player only -- it has no starters or point totals -- so declaring that here
# keeps the plan audit from calling a working filter unused.
APPLIED_ROW_FILTERS = {
    "player_box_scores": set(ROW_FILTER_KEYS),
    "injury_notes": {"person_id"},
}
CONFERENCES = {"East", "West"}
GROUP_KEYS = {"person_id", "opponent_team_id"}

SOURCES = {"game_details", "player_box_scores", "game_recaps", "injury_notes"}
GAME_SELECTS = {"first", "last", "all"}
AGG_OPS = {"none", "sum", "avg", "count", "count_games", "max", "min", "argmax",
           "argmin", "max_streak"}
INNER_OPS = {"sum", "count", "max", "min", "avg"}
MAX_RANK = 10

# Numeric columns a plan may aggregate over. Restricted to a whitelist so a
# malformed plan can never interpolate an arbitrary identifier into SQL.
BOX_FIELDS = {
    "points", "assists", "steals", "blocks", "turnovers", "seconds",
    "offensive_reb", "defensive_reb", "rebounds",
    "fg2_made", "fg2_attempted", "fg3_made", "fg3_attempted",
    "ft_made", "ft_attempted",
    # Shooting percentages. Aggregated as total makes over total attempts, not
    # as the mean of per-game percentages -- averaging percentages weights a
    # 1-for-1 night the same as a 10-for-20 one and gives the wrong answer.
    "fg3_pct", "ft_pct", "fg_pct", "fg2_pct",
}
GAME_FIELDS = {"margin", "home_points", "away_points", "total_points",
               # Relative to the subject team, so "points allowed" is the
               # opponent's score in each of that team's games.
               "points_for", "points_allowed"}
AGG_FIELDS = BOX_FIELDS | GAME_FIELDS

# The fixed vocabulary of values execute.py can produce. A plan's `outputs` maps
# these onto whatever field names the question's `return` block happens to use,
# which is how one code path serves questions whose wording and field names are
# not known in advance.
PRIMITIVES = {
    "agg",            # the aggregate value itself
    "n_games",        # distinct games in the result set
    "n_rows",         # rows in the result set
    "group_name",     # player name of the winning group in an argmax/argmin
    "score",          # "<winner>-<loser>", formatted from the table
    "winner_name",    # mascot of the winning team
    "opponent_name",  # mascot of the opponent, relative to the subject team
    "margin",         # winning margin of the selected game
    "game_date",      # ISO date of the selected game
    "n_teams",        # distinct teams a player appeared for
    "team_name",      # mascot(s) of the team(s) a player appeared for
    "n_players",      # distinct players among the cited rows
    "n_wins",         # wins for the subject team among the selected games
    "n_losses",       # losses for the subject team among the selected games
    "text",           # this field is stated only in a recap or injury note
}

# Which primitives can fill which declared type. Used to reject nonsense pairings
# such as mapping a "score" field (a string like "128-110") onto the numeric
# margin -- a mistake the model made unprompted, which would have shipped "18"
# as a final score.
NUMERIC_PRIMITIVES = {"agg", "n_games", "n_rows", "n_teams", "n_players",
                      "n_wins", "n_losses", "margin", "text"}
STRING_PRIMITIVES = {"score", "winner_name", "opponent_name", "group_name",
                     "game_date", "team_name", "text"}

# Primitives a prose source can legitimately supply. A recap or an injury note
# is not a table of numbers: there is no aggregate to compute and no player
# grouping to take a maximum over, so anything outside this set has to come from
# the text itself. Without this, a plan reading recaps would ask the executor for
# an "agg" that was never computed, and the question would abstain.
TEXT_SOURCE_PRIMITIVES = {"score", "winner_name", "opponent_name", "margin",
                          "game_date", "n_games", "text"}

# Injury notes are prose with structure around it: player_id and status are
# columns, so absences can be counted and ranked by player the same way box
# score rows can. Recaps have no such handle -- a recap names a player only in
# its sentences -- so they cannot.
NOTE_SOURCE_PRIMITIVES = TEXT_SOURCE_PRIMITIVES | {"group_name", "n_players", "agg"}


def text_primitives_for(source: str) -> set:
    """Primitives a prose source can supply without reading its prose.

    :param source: the plan's source table
    :return: the allowed primitive names
    """
    return NOTE_SOURCE_PRIMITIVES if source == "injury_notes" else TEXT_SOURCE_PRIMITIVES

PLAN_SPEC = """
{
  "source": "game_details | player_box_scores | game_recaps | injury_notes",
  "game_filters": {
    "team_id": int|null,            // subject team played in the game
    "opponent_team_id": int|null,   // the other team in the game
    "game_id": int|null,
    "date_from": "YYYY-MM-DD"|null,
    "date_to": "YYYY-MM-DD"|null,
    "outcome": "win"|"loss"|null,   // result for team_id
    "min_margin": int|null          // game decided by at least this many points
  },
  "game_select": "first|last|all",  // ordinal over the filtered games, by date
  "row_filters": {
    "person_id": int|null,
    "team_id": int|null,            // player's team in that game
    "exclude_team_id": int|null,    // players NOT on this team, i.e. "against X"
    "starter": true|false|null,
    "min_points": int|null
  },
  "aggregate": {
    "op": "none|sum|avg|count|count_games|max|min|argmax|argmin",
    "field": "points|assists|rebounds|margin|...",
    "group_by": "person_id"|null,   // required for argmax/argmin
    "inner": "sum|count|max|avg",   // how to combine within a group
    "rank": 1                       // 2 for "second-highest", 3 for "third", ...
  },
  "outputs": { "<return field>": "<primitive>" }
}
""".strip()

PRIMITIVE_HELP = """
agg = the aggregate value        n_games = number of distinct games
n_rows = number of rows          group_name = player name winning an argmax
score = "winner-loser"           winner_name = winning team mascot
opponent_name = opponent mascot  margin = winning margin
game_date = date of the game
text = the fact is stated only in a recap or an injury note
""".strip()

# Worked examples of the plan *shape*. The ids below (7001 for a player, 9001
# and 9002 for teams) are placeholders and belong to nobody: they are here to
# show where an id goes, not which id to use. Copying a real id out of an
# example would be a filter about the wrong team, and validate_plan drops any id
# that did not come from this question's own resolution anyway -- but an example
# carrying an id from the public question set is the wrong thing to show a model
# that is about to be asked something else entirely.
EXAMPLES = """
Q: "How many points did <Player> score in <Month> <Year>?"  return: points, games
{"source":"player_box_scores","game_filters":{"date_from":"<YYYY-MM-01>","date_to":"<YYYY-MM-31>"},
 "game_select":"all","row_filters":{"person_id":7001},
 "aggregate":{"op":"sum","field":"points","group_by":null,"inner":"sum"},
 "outputs":{"points":"agg","games":"n_games"}}

Q: "Which <Team> starter scored the most total points in games <Team> lost by 10+?"  return: player_name, points
{"source":"player_box_scores","game_filters":{"team_id":9001,"outcome":"loss","min_margin":10},
 "game_select":"all","row_filters":{"team_id":9001,"starter":true},
 "aggregate":{"op":"argmax","field":"points","group_by":"person_id","inner":"sum"},
 "outputs":{"player_name":"group_name","points":"agg"}}

Q: "How many points did <Player> score in <Team>'s season opener?"  return: points
{"source":"player_box_scores","game_filters":{"team_id":9001},
 "game_select":"first","row_filters":{"person_id":7001},
 "aggregate":{"op":"sum","field":"points","group_by":null,"inner":"sum"},
 "outputs":{"points":"agg"}}

Q: "What was <Team>'s largest margin of defeat, who beat them, and the score?"  return: opponent, margin, score
{"source":"game_details","game_filters":{"team_id":9001,"outcome":"loss"},
 "game_select":"all","row_filters":{},
 "aggregate":{"op":"max","field":"margin","group_by":null,"inner":"max"},
 "outputs":{"opponent":"opponent_name","margin":"margin","score":"score"}}

Q: "Who led <Team> in rebounds against <Team2> on <Date>?"  return: player_name, rebounds
{"source":"player_box_scores","game_filters":{"team_id":9001,"opponent_team_id":9002,
 "date_from":"<YYYY-MM-DD>","date_to":"<YYYY-MM-DD>"},
 "game_select":"all","row_filters":{"team_id":9001},
 "aggregate":{"op":"argmax","field":"rebounds","group_by":"person_id","inner":"sum"},
 "outputs":{"player_name":"group_name","rebounds":"agg"}}
""".strip()


def _first_number(patterns: list[str], haystack: str) -> int | None:
    """Return the first number any of `patterns` captures.

    :param patterns: regexes with one capturing group
    :param haystack: text to search
    :return: the captured integer, or None
    """
    for pattern in patterns:
        m = re.search(pattern, haystack, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


BEAT_VERBS = r"beat|beats|defeated|defeat|downed|topped|knocked off|got past"


def _beat_direction(question_text: str, teams: list[dict]) -> str | None:
    """Work out who did the beating, from word order around the verb.

    "Which team did the Outriders beat most often" and "which team beat the
    Outriders most often" name the same two sides and mean opposite things. The
    subject team's position relative to the verb is what separates them, and
    getting it wrong returns the team they lost to most instead of the one they
    beat most.

    :param question_text: the raw question
    :param teams: resolved teams; the first is the subject
    :return: "win", "loss", or None when no beating verb applies to the subject
    """
    if not teams:
        return None
    subject = teams[0]
    labels = [f"{subject['city']} {subject['name']}", subject["name"],
              subject["city"], subject["abbreviation"]]
    for label in labels:
        escaped = re.escape(label)
        # "... the Outriders beat ..." -- subject is doing the beating.
        if re.search(rf"\b{escaped}\b(?:\s+\w+){{0,2}}?\s+(?:{BEAT_VERBS})\b",
                     question_text, re.IGNORECASE):
            return "win"
        # "... beat the Outriders ..." -- subject is being beaten.
        if re.search(rf"\b(?:{BEAT_VERBS})\s+(?:the\s+)?{escaped}\b",
                     question_text, re.IGNORECASE):
            return "loss"
    return None


def infer_rank(question_text: str) -> int:
    """Read the rank a question asks for: 1 for a superlative, N for an ordinal.

    :param question_text: the raw question
    :return: the rank, defaulting to 1
    """
    if re.search(r"\brunner[-\s]?up\b", question_text, re.IGNORECASE):
        return 2
    alternation = "|".join(ORDINAL_WORDS)
    m = re.search(rf"\b({alternation})\b[-\s]*(?:{RANK_SUPERLATIVES})",
                  question_text, re.IGNORECASE)
    return ORDINAL_WORDS[m.group(1).lower()] if m else 1


def infer_field(question_text: str, source: str) -> str | None:
    """Read which statistic a question is about from its wording.

    Left to the model, the field defaults to points regardless of what was
    asked, so a rebounding question gets a scoring answer -- fluent, cited, and
    about the wrong column. The question names the stat, so the question decides
    it.

    :param question_text: the raw question
    :param source: the plan's source table
    :return: a field name from AGG_FIELDS, or None if the wording names no stat
    """
    table = GAME_FIELD_WORDS if source == "game_details" else BOX_FIELD_WORDS
    for pattern, field in table:
        if re.search(pattern, question_text, re.IGNORECASE):
            return field
    return None


def infer_aggregate(question_text: str) -> dict:
    """Read the aggregate shape the question's wording implies.

    Covers the two things the plan DSL could express but the model consistently
    failed to ask for: an average rather than a total, and a rank other than the
    top one. Both were previously answered with a confident value for a
    different question -- a season total offered as a per-game average, and the
    leading scorer offered as the second-highest.

    :param question_text: the raw question
    :return: aggregate keys to apply over the model's choice
    """
    overrides: dict = {
        "rank": infer_rank(question_text),
        # Direction is authoritative. The model reads "fewest" and reaches for
        # argmax anyway, which returns the exact opposite player.
        "descending": not re.search(FEWEST_WORDS, question_text, re.IGNORECASE),
        "single_game": bool(re.search(SINGLE_GAME_WORDS, question_text, re.IGNORECASE)),
        "by_team": bool(re.search(TEAM_SUBJECT_WORDS, question_text, re.IGNORECASE))
                   or bool(re.search(WHO_BEAT_WORDS, question_text, re.IGNORECASE)
                           and re.search(FREQUENCY_WORDS, question_text, re.IGNORECASE)),
    }
    if re.search(AVERAGE_WORDS, question_text, re.IGNORECASE):
        overrides["average"] = True
    return overrides


def infer_filters(question_text: str, resolved: dict, source: str) -> tuple[dict, dict, str]:
    """Read the filters a question states outright, without asking the model.

    Covers the parts of a plan that are stated in plain English and are therefore
    recoverable by pattern: a stat threshold ("30 or more points"), a margin
    ("lost by 10 or more"), an outcome ("their loss against"), an ordinal
    ("season opener"), whether starters are being asked about, and whether the
    players in question are the subject team's own or the ones facing them.

    That last distinction is the one worth spelling out. "Which Outriders starter
    scored the most" and "which player scored 30+ against the Outriders" both
    resolve to the same single team, but one is asking about that team's roster
    and the other about everyone else's. The difference is the word "against",
    so that is what gets matched.

    :param question_text: the raw question
    :param resolved: output of resolve.resolve_question()
    :param source: the plan's source table
    :return: (game_filters, row_filters, game_select) to apply over the model's plan
    """
    game_filters: dict = {}
    row_filters: dict = {}
    game_select = None

    # Margins are read first and their wording removed, because "lost by 10 or
    # more points" also satisfies the per-player threshold patterns. Leaving it
    # in made a question about a 10-point *margin* filter out every player who
    # scored under 10, which quietly changes the answer.
    # Dates are authoritative. resolve_dates has already read the question;
    # letting the model's range stand instead meant "after the All-Star break"
    # came back with the whole season, because the model simply omitted it.
    dates = resolved["dates"]
    game_filters["date_from"] = dates["date_from"].isoformat()
    game_filters["date_to"] = dates["date_to"].isoformat()

    if re.search(HOME_WORDS, question_text, re.IGNORECASE):
        game_filters["venue"] = "home"
    elif re.search(AWAY_WORDS, question_text, re.IGNORECASE):
        game_filters["venue"] = "away"
    else:
        game_filters["venue"] = None

    remaining = question_text
    margin = _first_number(MARGIN_PATTERNS, question_text)
    if margin is not None:
        game_filters["min_margin"] = margin
        for pattern in MARGIN_PATTERNS:
            remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)

    # A ceiling on the margin, read and stripped the same way, so that the
    # "3 points" in "decided by 3 points or fewer" cannot also be picked up as
    # a per-player scoring threshold.
    tight = _first_number(MAX_MARGIN_INCLUSIVE, question_text)
    if tight is None:
        strict = _first_number(MAX_MARGIN_EXCLUSIVE, question_text)
        tight = None if strict is None else strict - 1
    game_filters["max_margin"] = tight
    if tight is not None:
        for pattern in MAX_MARGIN_INCLUSIVE + MAX_MARGIN_EXCLUSIVE:
            remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)

    # Thresholds, per stat. Set or clear, like outcome below: a threshold the
    # question never stated is a condition nobody asked for.
    # Built per stat from an explicit noun, not by patching the word "points"
    # into the shared patterns: one of those ends in "point" (singular), which
    # no substitution touched, so every stat matched "30 or more points" and a
    # scoring question picked up a phantom 30-rebound filter.
    for stat, key in (("points?", "min_points"), ("rebounds?|boards", "min_rebounds"),
                      ("assists?|dimes", "min_assists")):
        inclusive = [
            rf"(\d+)\s*\+?\s*or\s+more\s+(?:{stat})\b",
            rf"at\s+least\s+(\d+)\s+(?:{stat})\b",
            rf"(\d+)\+\s+(?:{stat})\b",
        ]
        exclusive = [
            rf"more\s+than\s+(\d+)\s+(?:{stat})\b",
            rf"over\s+(\d+)\s+(?:{stat})\b",
            rf"above\s+(\d+)\s+(?:{stat})\b",
        ]
        value = _first_number(inclusive, remaining)
        if value is None:
            strict = _first_number(exclusive, remaining)
            value = None if strict is None else strict + 1
        row_filters[key] = value

    # A threshold whose stat is named by the verb instead of by a noun. "How
    # many times did Tavian Belcourt score 30 or more" states a floor as plainly
    # as "30 or more points" does, but every pattern above needs the noun, so
    # the threshold was dropped and the answer came back as all 64 games he
    # played. The stat comes from the same reader the rest of the pipeline uses.
    #
    # The "or more" marker is required, and the lookahead keeps this off any
    # phrase a named stat already claimed: without them "scored 2143 points"
    # would read as a 2143-point floor and match nothing.
    THRESHOLD_KEYS = {"points": "min_points", "rebounds": "min_rebounds",
                      "assists": "min_assists"}
    if all(row_filters.get(k) is None for k in THRESHOLD_KEYS.values()):
        stat_noun = r"(?!\s*(?:points?|rebounds?|boards|assists?|dimes))"
        bare = _first_number([rf"(\d+)\s*\+{stat_noun}",
                              rf"(\d+)\s*\+?\s*or\s+(?:more|better){stat_noun}",
                              rf"at\s+least\s+(\d+){stat_noun}"], remaining)
        key = THRESHOLD_KEYS.get(infer_field(question_text, "player_box_scores"))
        if bare is not None and key is not None:
            row_filters[key] = bare

    # "Scored no points" is a ceiling, not a floor. Without this the phrase set
    # no filter at all and the question counted every game the player appeared
    # in -- 68 games as the answer to "how many games did he score none".
    scoreless = re.search(r"\b(?:no|zero|0)\s+points\b|\bscoreless\b|\bfailed to score\b",
                          question_text, re.IGNORECASE)
    row_filters["max_points"] = 0 if scoreless else None
    if scoreless:
        row_filters["min_points"] = None

    # Outcome is set *or cleared*, never just set. A model that invents
    # "outcome": "win" on a question that says nothing about winning silently
    # discards half the season, and there is no later step that could notice.
    # The question's own wording is the authority on this one.
    directional = _beat_direction(question_text, resolved["teams"])
    if directional is not None:
        game_filters["outcome"] = directional
    elif re.search(LOSS_WORDS, question_text, re.IGNORECASE):
        game_filters["outcome"] = "loss"
    elif re.search(WIN_WORDS, question_text, re.IGNORECASE):
        game_filters["outcome"] = "win"
    else:
        game_filters["outcome"] = None

    # Authoritative, like outcome and the row filters: an ordinal the question
    # never used must not survive from the model's plan. An invented
    # game_select: "first" silently narrowed a season-long average down to the
    # opener and reported that one game's points as the average.
    nth = re.search(NTH_GAME_PATTERN, question_text, re.IGNORECASE)
    if re.search(FIRST_WORDS, question_text, re.IGNORECASE):
        game_select = "first"
    elif re.search(LAST_WORDS, question_text, re.IGNORECASE):
        game_select = "last"
    elif nth:
        game_select = int(nth.group(1)) if nth.group(1) else NTH_WORDS[nth.group(2).lower()]
    else:
        game_select = "all"

    # Bench and starter are the two halves of a roster, so "who led the bench"
    # must not be answered with the starter who led the team.
    if (re.search(BENCH_WORDS, question_text, re.IGNORECASE)
            or re.search(NOT_STARTED_WORDS, question_text, re.IGNORECASE)):
        row_filters["starter"] = False
    elif (re.search(STARTER_WORDS, question_text, re.IGNORECASE)
            or re.search(STARTED_VERB_WORDS, question_text, re.IGNORECASE)):
        row_filters["starter"] = True
    else:
        row_filters["starter"] = None

    if re.search(EAST_WORDS, question_text, re.IGNORECASE):
        game_filters["opponent_conference"] = "East"
    elif re.search(WEST_WORDS, question_text, re.IGNORECASE):
        game_filters["opponent_conference"] = "West"
    else:
        game_filters["opponent_conference"] = None

    position = None
    for pattern, code in POSITION_WORDS:
        if re.search(pattern, question_text, re.IGNORECASE):
            position = code
            break
    row_filters["position"] = position

    # A double-double is any two categories in double figures, which is a
    # condition on the row rather than a threshold on one stat.
    row_filters["triple_double"] = True if re.search(
        TRIPLE_DOUBLE_WORDS, question_text, re.IGNORECASE) else None
    row_filters["double_double"] = True if (
        re.search(DOUBLE_DOUBLE_WORDS, question_text, re.IGNORECASE)
        and not row_filters["triple_double"]) else None

    # Whose players are being asked about. team_id and exclude_team_id are
    # mutually exclusive -- a row cannot both be and not be on a team -- so
    # whichever applies is asserted and the other is cleared. Leaving the
    # model's choice in place alongside produced `team_id = 486 AND team_id <>
    # 486`, which matches nothing and looked exactly like a missing player.
    teams = resolved["teams"]
    players = resolved["players"]
    exact_players = [p for p in players if p["exact"]]
    named_player = len(exact_players) == 1

    # "Games played without X" constrains the game set by a player's absence,
    # the inverse of every other player filter. Only meaningful over games,
    # which is also what keeps it clear of "why did X miss this game" -- that
    # question reads an injury note and is about the player, not the schedule.
    game_filters["exclude_person_id"] = None
    if (source == "game_details" and exact_players
            and re.search(WITHOUT_WORDS, question_text, re.IGNORECASE)):
        game_filters["exclude_person_id"] = exact_players[0]["player_id"]

    # Two players named in one question is a comparison ("who scored more, X or
    # Y"). Restricting to exactly those two and taking the argmax answers it;
    # without this the filter was dropped entirely and the question was silently
    # answered about whoever led the game.
    row_filters["person_ids"] = ([p["player_id"] for p in exact_players]
                                 if len(exact_players) > 1 else None)

    # The named player is authoritative, like every other resolved entity. It
    # used to be set only in the prefill skeleton, so whenever the model
    # returned a usable plan of its own the filter vanished -- and a question
    # about one player's absence was answered from every note in the game.
    row_filters["person_id"] = (exact_players[0]["player_id"]
                                if named_player and game_filters.get("exclude_person_id") is None
                                else None)

    # Subject and opponent follow the order the question names them, which is
    # how these questions are written ("for Saltmere ... against LMD").
    # Leaving the model's assignment in place let it swap the two, and asking
    # for the winner's game under the loser's name selected no games at all.
    if teams:
        game_filters["team_id"] = teams[0]["team_id"]
    if len(teams) > 1:
        game_filters["opponent_team_id"] = teams[1]["team_id"]

    # "Who led all scorers" spans both benches. Scoping it to the subject team
    # answers a narrower question than the one asked, and quietly drops the
    # opponent's leading scorer from consideration.
    both_teams = re.search(r"\ball scorers\b|\beither team\b|\bboth teams\b"
                           r"|\banyone in the game\b|\bany player in\b",
                           question_text, re.IGNORECASE)

    if source == "player_box_scores" and both_teams:
        row_filters["team_id"] = None
        row_filters["exclude_team_id"] = None
    elif source == "player_box_scores":
        team_key = exclude_key = None
        if named_player:
            # person_id already pins the rows; a team filter on top can only
            # exclude the player's own line when the team named is the opponent.
            pass
        elif teams:
            subject = teams[0]
            labels = [f"{subject['city']} {subject['name']}", subject["name"],
                      subject["city"], subject["abbreviation"]]
            # "the" is optional, because questions say "against Sablecrest" as
            # readily as "against the Sablecrest Outriders" -- and requiring it made
            # the first form count that team's *own* players. "on" is included
            # for "dropped 30+ on the Gannets", which means the same thing.
            facing = any(
                re.search(rf"\b(?:against|versus|vs\.?|facing|opposing|on)\s+"
                          rf"(?:the\s+)?{re.escape(lbl)}\b",
                          question_text, re.IGNORECASE)
                for lbl in labels
            )
            if facing:
                exclude_key = subject["team_id"]
            else:
                team_key = subject["team_id"]
        row_filters["team_id"] = team_key
        row_filters["exclude_team_id"] = exclude_key

    return game_filters, row_filters, game_select


def source_for(question: dict) -> str:
    """Pick the table a question should be read from, out of what it declares.

    The `return.evidence` block names the tables a question is allowed to cite,
    which makes it the authority on where the answer lives -- more reliable than
    the model, which on one run read box scores for a question that declared
    injury notes and answered "how many games did X miss" with the number he
    played.

    Text tables take priority. A question that declares a recap or an injury
    note is telling us the answer is written down in prose; the structured
    tables then serve to pick *which* passage to read, and to supply the
    surrounding facts (score, opponent) once it has been read.

    :param question: one entry from questions/questions.json
    :return: the source table name
    """
    tables = {s.get("table") for s in question["return"].get("evidence", [])}
    if "game_recaps" in tables:
        return "game_recaps"
    if "injury_notes" in tables:
        return "injury_notes"
    if "player_box_scores" in tables:
        return "player_box_scores"
    return "game_details"


def blank_plan() -> dict:
    """Return an empty, structurally valid plan.

    :return: plan dict with every key present and no filters set
    """
    return {
        "source": "player_box_scores",
        "game_filters": {},
        "game_select": "all",
        "row_filters": {},
        "aggregate": {"op": "none", "field": "points", "group_by": None,
                      "inner": "sum", "rank": 1},
        "outputs": {},
    }


def prefill_plan(question: dict, resolved: dict) -> dict:
    """Derive the part of the plan that follows from entity resolution alone.

    Anything that can be decided deterministically is decided here, so the model
    has less room to get it wrong: an unambiguously resolved player becomes a
    person_id filter, an explicit calendar date becomes a date range, and the
    source table follows from what the question's `return.evidence` block
    declares. The model refines this skeleton rather than building one.

    When a question names exactly two teams they are the two sides of a game, so
    both are filled in -- subject first, opponent second, following the order
    they are mentioned (resolve_teams sorts by position). "The Outriders' game
    against the Coyotes" makes the Outriders the subject, which is what decides
    whose loss it is and which mascot "opponent" refers to. The model can still
    swap them if the wording runs the other way.

    :param question: one entry from questions/questions.json
    :param resolved: output of resolve.resolve_question()
    :return: a partially filled plan
    """
    plan = blank_plan()
    plan["source"] = source_for(question)

    dates = resolved["dates"]
    plan["game_filters"]["date_from"] = dates["date_from"].isoformat()
    plan["game_filters"]["date_to"] = dates["date_to"].isoformat()

    players = resolved["players"]
    if len(players) == 1 and players[0]["exact"]:
        plan["row_filters"]["person_id"] = players[0]["player_id"]

    teams = resolved["teams"]
    if teams:
        plan["game_filters"]["team_id"] = teams[0]["team_id"]
    if len(teams) == 2:
        plan["game_filters"]["opponent_team_id"] = teams[1]["team_id"]

    game_filters, row_filters, game_select = infer_filters(
        question["question"], resolved, plan["source"])
    _merge_inferred(plan["game_filters"], game_filters)
    _merge_inferred(plan["row_filters"], row_filters)
    if game_select:
        plan["game_select"] = game_select

    return plan


def _merge_inferred(target: dict, updates: dict) -> None:
    """Apply inferred filters, treating None as "the question did not ask for this".

    Inference is authoritative in both directions. A value it found is written;
    a key it explicitly resolved to None is removed, so a condition the model
    invented cannot survive into the query.

    :param target: the plan's filter dict, modified in place
    :param updates: inferred values, where None means clear
    """
    for key, value in updates.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


# Field names that mean "how many games", as opposed to "how much of a stat".
# Used to decide whether grouping should count rows or total them.
COUNT_FIELD_WORDS = r"game|count|appearance|night|outing"


def enforce_output_coherence(plan: dict, question: dict) -> dict:
    """Make the aggregate consistent with what the outputs ask for.

    A plan that asks for `group_name` is asking "which player" -- and only
    argmax/argmin produce a winning player. The model does not always connect
    the two: on one run it paired a request for a player name with
    `op: "count_games"`, which counts games and names nobody, so a question that
    had a perfectly good answer abstained instead.

    Rather than trusting the model to keep the two halves in sync, the op is
    derived from the outputs, which are the half that is checked against the
    question's declared `return` block.

    :param plan: the plan, modified in place
    :param question: one entry from questions/questions.json
    :return: the same plan
    """
    outputs = plan["outputs"]
    agg = plan["aggregate"]

    # A field named after a statistic means that statistic, so the aggregate has
    # to be over it rather than a count of rows. Counting looks harmless until
    # the row set is a single game: "how many points did Stroud score in the
    # Outriders' tenth game" returned 1 -- the number of box score rows -- for a
    # game in which he scored 18.
    stat_fields = [f for f, p in outputs.items() if p == "agg" and f in AGG_FIELDS]
    if stat_fields and agg["op"] in ("count", "count_games", "none"):
        agg["op"] = "sum"
        agg["field"] = stat_fields[0]

    # Counting distinct teams or players needs every qualifying row cited, so an
    # aggregate that narrows to a single best row would undercount. Nothing is
    # being ranked here, so the op is replaced with one that keeps them all.
    counting_primitives = {"n_teams", "team_name", "n_players", "n_wins", "n_losses"}
    if (counting_primitives & set(outputs.values())
            and agg["op"] in ("max", "min", "argmax", "argmin")):
        agg["op"] = "count"
        agg["group_by"] = None

    # A win-loss record needs the wins *and* the losses. "Win-loss record"
    # contains the word "win", which set an outcome filter and left the losses
    # at zero -- a record of 64-0.
    if {"n_wins", "n_losses"} <= set(outputs.values()):
        plan["game_filters"].pop("outcome", None)

    if "group_name" not in outputs.values():
        return plan

    if agg["op"] not in ("argmax", "argmin"):
        agg["op"] = ("argmin" if re.search(FEWEST_WORDS, question["question"], re.IGNORECASE)
                     else "argmax")
    if agg["group_by"] is None:
        agg["group_by"] = ("opponent_team_id"
                           if re.search(TEAM_SUBJECT_WORDS, question["question"], re.IGNORECASE)
                           else "person_id")

    # Leave a combiner the question asked for. "Average", and the per-game
    # extremes behind "the fewest points in a game", were both inferred from
    # wording; overwriting them here reverted those questions to season totals.
    if agg["inner"] in ("avg", "min", "max"):
        return plan

    # Re-apply the single-game reading now that the op is settled. It is decided
    # before this function runs, but only took effect when the model had already
    # produced an argmax -- so "the fewest points in a game" kept coming back as
    # the lowest season total whenever the model had said "min" instead.
    if re.search(SINGLE_GAME_WORDS, question["question"], re.IGNORECASE):
        agg["inner"] = "max" if agg["op"] == "argmax" else "min"
        return plan

    counting = any(re.search(COUNT_FIELD_WORDS, field, re.IGNORECASE)
                   and primitive in ("agg", "n_games")
                   for field, primitive in outputs.items())
    agg["inner"] = "count" if counting else "sum"
    # When the ranking counts games, that count is the answer to "how many
    # games": the winner's own tally. n_games is the number of distinct games
    # among the cited rows, which matches for a lone winner and becomes the
    # union for a tie -- "which Outriders player started the most games" named
    # Ferreira and Stroud, 69 starts each, and reported 79.
    if counting:
        for field, primitive in outputs.items():
            if primitive == "n_games" and re.search(COUNT_FIELD_WORDS, field, re.IGNORECASE):
                outputs[field] = "agg"
    return plan


def _apply_aggregate_overrides(plan: dict, question_text: str) -> None:
    """Apply the rank and averaging the question's wording calls for.

    An average is expressed as an inner combiner so it composes with grouping:
    ungrouped it is the plain mean of the rows, and grouped it makes
    "which player averaged the most" mean what it says rather than "who totalled
    the most".

    :param plan: the plan, modified in place
    :param question_text: the raw question
    """
    overrides = infer_aggregate(question_text)
    agg = plan["aggregate"]
    agg["rank"] = overrides["rank"]
    field = infer_field(question_text, plan["source"])
    if field is not None:
        agg["field"] = field

    # A single-game margin extreme. Asserted before the direction and grouping
    # rules below so that "worst loss" is the biggest defeat rather than the
    # narrowest, and "against which opponent did they post their largest margin"
    # picks one game rather than totalling margins per opponent.
    if re.search(MARGIN_EXTREME_WORDS, question_text, re.IGNORECASE):
        agg["field"], agg["op"] = "margin", "max"
        agg["group_by"] = None
        overrides["descending"] = True
        overrides["by_team"] = False

    # "Closest" and "blowout" both name the margin, pointed opposite ways.
    if re.search(CLOSEST_WORDS, question_text, re.IGNORECASE):
        agg["field"], agg["op"] = "margin", "min"
        overrides["descending"] = False
    elif re.search(BLOWOUT_WORDS, question_text, re.IGNORECASE):
        agg["field"], agg["op"] = "margin", "max"
        overrides["descending"] = True
    if overrides.get("average"):
        if agg["op"] in ("argmax", "argmin"):
            agg["inner"] = "avg"
        else:
            agg["op"] = "avg"
    # A rank above the top only means something for an ordered aggregate; a sum
    # has no second place.
    if agg["rank"] > 1 and agg["op"] in ("none", "sum", "avg", "count", "count_games"):
        agg["op"] = "argmax"
        agg["group_by"] = "person_id"

    # Point the comparison the way the question points it.
    descending = overrides["descending"]
    if agg["op"] in ("argmax", "argmin"):
        agg["op"] = "argmax" if descending else "argmin"
    elif agg["op"] in ("max", "min"):
        agg["op"] = "max" if descending else "min"

    # "In a game" compares single performances rather than season totals.
    if overrides["single_game"] and agg["op"] in ("argmax", "argmin"):
        agg["inner"] = "max" if descending else "min"

    # A streak question is about consecutive results, which no group-and-
    # aggregate can express; left alone it came back as "1".
    if re.search(STREAK_WORDS, question_text, re.IGNORECASE):
        agg["op"] = "max_streak"
        agg["group_by"] = None

    # "Which team ..." ranks opponents, not players. Forced regardless of the op
    # the model chose: without this, "which team did they beat most often" ran
    # as a plain max over margin and named whoever they beat worst once.
    if overrides["by_team"]:
        agg["op"] = "argmax" if descending else "argmin"
        agg["group_by"] = "opponent_team_id"
        if re.search(r"\bmost often\b|\bhow many times\b|\bmost frequently\b|\bmost\b",
                     question_text, re.IGNORECASE):
            agg["inner"] = "count"


def apply_inferred(plan: dict, question: dict, resolved: dict) -> dict:
    """Re-assert the deterministically inferred filters over a model's plan.

    Applied *after* validation, so the model can add what it understood but
    cannot remove what the question states outright. The model is better than a
    regex at reading which team is the subject and what kind of aggregate is
    wanted; it is worse at remembering that "30 or more points" was a condition.
    Each side wins where it is stronger.

    :param plan: a validated plan
    :param question: one entry from questions/questions.json
    :param resolved: output of resolve.resolve_question()
    :return: the plan with inferred filters applied
    """
    # The declared evidence tables decide where to read from, not the model.
    plan["source"] = source_for(question)

    game_filters, row_filters, game_select = infer_filters(
        question["question"], resolved, plan["source"])
    _merge_inferred(plan["game_filters"], game_filters)
    _merge_inferred(plan["row_filters"], row_filters)
    if game_select:
        plan["game_select"] = game_select
    _apply_aggregate_overrides(plan, question["question"])

    # If the question named a team, the games must involve it. This is a
    # gap-fill, not an override: when the model supplied a team it keeps its
    # reading, since deciding which of two named teams is the subject is a
    # judgement call it makes better than a rule. But when the model dropped the
    # team entirely, "against the Outriders on April 12" widened to every game
    # played that day.
    if plan["game_filters"].get("team_id") is None and resolved["teams"]:
        plan["game_filters"]["team_id"] = resolved["teams"][0]["team_id"]

    # An outcome filter is only meaningful relative to a team, and select_games
    # returns nothing without one.
    if plan["game_filters"].get("outcome") and plan["game_filters"].get("team_id") is None:
        plan["game_filters"].pop("outcome", None)
    return plan


def build_prompt(question: dict, resolved: dict, prefill: dict) -> str:
    """Build the planning prompt.

    Gives the model the resolved entities as an explicit menu of ids, so filling
    in the plan is a selection task rather than open-ended generation. The
    question's own `return` block is included because it defines the output field
    names the plan must map onto -- that is what lets the same prompt serve
    questions this code has never seen.

    :param question: one entry from questions/questions.json
    :param resolved: output of resolve.resolve_question()
    :param prefill: skeleton from prefill_plan()
    :return: prompt text for ollama_generate
    """
    teams = [{"mascot": t["name"], "team_id": t["team_id"]} for t in resolved["teams"]]
    players = [{"name": p["full_name"], "person_id": p["player_id"]} for p in resolved["players"]]
    return (
        "You translate a basketball question into a JSON query plan. "
        "You never compute values; a database executes your plan.\n\n"
        f"PLAN FORMAT:\n{PLAN_SPEC}\n\n"
        f"PRIMITIVES you may use in \"outputs\":\n{PRIMITIVE_HELP}\n\n"
        f"EXAMPLES:\n{EXAMPLES}\n\n"
        "RULES:\n"
        "- Use ONLY the ids listed under ENTITIES. Never invent an id.\n"
        "- \"outputs\" must contain exactly the RETURN FIELDS listed below, except \"answerable\".\n"
        "- Use game_select \"first\" for a season opener / earliest game, \"last\" for the most recent.\n"
        "- For \"which player ...\" use argmax with group_by \"person_id\".\n"
        "- inner \"count\" counts qualifying games per player; inner \"sum\" totals the stat.\n"
        "- Respond with the JSON plan only.\n\n"
        f"ENTITIES:\n  teams: {json.dumps(teams)}\n  players: {json.dumps(players)}\n"
        f"  dates: {prefill['game_filters']['date_from']} to {prefill['game_filters']['date_to']}\n\n"
        f"RETURN FIELDS: {json.dumps(question['return'])}\n\n"
        f"STARTING POINT (refine it):\n{json.dumps(prefill)}\n\n"
        f"QUESTION: {question['question']}\n\nPLAN JSON:"
    )


# How a return-block field name maps onto an executor primitive when the model
# gives us nothing usable. Field names are the stable part of the contract --
# questions.json declares them and the output is checked against them -- so mapping
# by name generalizes to new questions in a way that mapping by question would not.
FIELD_PRIMITIVES = {
    "games": "n_games",
    "score": "score",
    "winner": "winner_name",
    "opponent": "opponent_name",
    "margin": "margin",
    "date": "game_date",
    "game_date": "game_date",
    "player_name": "group_name",
    "player": "group_name",
    "teams": "n_teams",
    "team": "team_name",
    "team_name": "team_name",
    "players": "n_players",
    "wins": "n_wins",
    "losses": "n_losses",
}


def default_outputs(question: dict, plan: dict) -> dict:
    """Map a question's return fields onto primitives without the model.

    The fallback for when planning fails outright. Names that describe a game
    fact map to that fact; a player name maps to the winner of an argmax; a
    remaining numeric field is the aggregate; and anything left over on a
    question that declares a prose table must be coming from the text.

    :param question: one entry from questions/questions.json
    :param plan: the plan being completed, whose aggregate may be adjusted
    :return: a field -> primitive mapping
    """
    declares_text = any(s.get("table") in ("game_recaps", "injury_notes")
                        for s in question["return"].get("evidence", []))
    outputs: dict = {}
    for field, type_name in question["return"].items():
        if field in ("answerable", "evidence"):
            continue
        if field in FIELD_PRIMITIVES:
            outputs[field] = FIELD_PRIMITIVES[field]
        elif declares_text:
            outputs[field] = "text"
        elif type_name in ("int", "float", "number"):
            # A bare numeric field is the thing being aggregated: "points",
            # "rebounds", "deficit".
            outputs[field] = "agg"
        else:
            # No sensible mapping. Leaving the field unmapped makes the caller
            # abstain, which is the whole point: the previous catch-all sent
            # every unrecognized field to "agg", so "how many teams did X play
            # for" was answered with his game count -- a real number, correctly
            # computed, for a question nobody asked.
            continue

        # A prose source cannot supply a computed value, so anything it cannot
        # produce must be coming from the passage. "Who hit the go-ahead basket"
        # maps to a player name, but that name is written in the recap, not
        # derived from an argmax over the box score.
        if (plan["source"] in ("game_recaps", "injury_notes")
                and outputs[field] not in text_primitives_for(plan["source"])):
            outputs[field] = "text"

        # Drop a mapping whose type cannot match the declared field, rather than
        # coercing a number into a name.
        allowed = (NUMERIC_PRIMITIVES if type_name in ("int", "float", "number")
                   else STRING_PRIMITIVES)
        if outputs[field] not in allowed:
            outputs.pop(field, None)

    # "Which player ..." is an argmax over players; anything else defaults to a
    # straight total of the stat being asked about.
    if "group_name" in outputs.values():
        plan["aggregate"]["group_by"] = "person_id"
        plan["aggregate"]["op"] = "argmax"
        plan["aggregate"]["inner"] = "count" if outputs.get("games") == "n_games" else "sum"
    elif plan["aggregate"]["op"] == "none":
        plan["aggregate"]["op"] = "sum"
    return outputs


def _clean_filters(raw, allowed: set) -> dict:
    """Keep only recognized filter keys with non-null values.

    :param raw: the model's filter object, not trusted to be a dict
    :param allowed: the key names valid in this filter scope
    :return: a filter dict containing only allowed, non-null entries
    """
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in allowed and v is not None}


# A question asking which *person* did something. The verb list is explicit so
# that "who won" and "who beat them" -- which ask about a team -- stay out of it.
PERSON_QUESTION = (
    r"\bwhich\b(?:\s+[\w'-]+){0,4}?\s+"
    r"(?:player|players|starter|starters|scorer|scorers|guard|guards"
    r"|forward|forwards|cent(?:er|re)s?|reserve|reserves)\b"
    r"|\bwho\s+(?:led|leads|scored|scores|hit|made|had|has|is|was|were"
    r"|missed|misses|averaged|averages|grabbed|dished|played|dropped|drops"
    r"|put|poured|posted|notched|recorded|tallied|piled|shot|finished"
    r"|pulled|collected|contributed|supplied|provided)\b"
    # "Came up with the go-ahead" is a person; "came out on top" is a team, and
    # matching the bare verb sent two score questions to a player shape.
    r"|\bwho came up with\b"
)
# Primitives that can actually name a person.
NAMING_PRIMITIVES = {"group_name", "text"}

# A question that counts things rather than totalling a statistic. The noun has
# to be a count noun -- "how many points" totals a stat and is a different
# question -- and a couple of adjectives may sit in front of it, since "how many
# home games" counts games just as plainly as "how many games".
COUNT_QUESTION = (r"\b(?:how many|number of)\s+(?:[\w'-]+\s+){0,2}?"
                  r"(?:games?|teams?|players?|wins?|losses|times|nights?)\b")
# Primitives that can actually produce a count.
COUNTING_PRIMITIVES = {"n_games", "n_rows", "n_teams", "n_players",
                       "n_wins", "n_losses"}

# Concepts that belong to a team rather than to any one player, so a plan
# reading player box scores has misread the question.
TEAM_CONCEPT = (r"\brecord\b|\bwin-loss\b|\bwins and losses\b|\bstandings?\b"
                r"|\bbest record\b|\bworst record\b")

# Vocabulary that belongs to the questions themselves rather than to anybody's
# name. Most of it is reused from the patterns above, so a word added for a
# filter or a statistic counts here automatically; this set only covers the
# ordinary English a basketball question is written in.
QUESTION_WORDS = {
    "opener", "openers", "opening", "finale", "closing", "earliest", "latest",
    "night", "nights", "matchup", "matchups", "starter", "starters", "bench",
    "start", "starts", "started",
    "reserve", "reserves", "lineup", "lineups", "rotation", "roster", "rookie",
    "play", "plays", "played", "playing", "appeared", "appearance",
    "appearances", "basket", "baskets", "shot", "shots", "make", "makes",
    "made", "attempt", "attempts", "attempted", "put", "poured", "posted",
    "notched", "tallied", "piled", "finished", "collected", "contributed",
    "supplied", "provided", "dropped", "drops", "grab", "grabbed", "dish",
    "dished", "pull", "pulled", "hit", "get", "got", "give", "gave", "came",
    "come", "back", "day", "days", "week", "weeks", "month", "months", "year",
    "years", "time", "times", "ago", "later", "earlier", "stat", "stats",
    "statistics", "line", "box", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "please", "me", "my", "we", "us",
    "did", "were", "been", "being", "be", "let", "know", "many", "long",
    "ever", "never", "still", "yet", "again", "once", "very", "really",
    # Determiners and comparatives. A league-wide question is allowed to name
    # nobody, so one unlisted word here is the difference between answering
    # "how many different players scored 30 or more" and refusing it.
    "different", "distinct", "unique", "individual", "separate", "various",
    "overall", "combined", "altogether", "whole", "entire", "full", "single",
    "multiple", "several", "few", "better", "worse", "best", "worst",
    "career", "regular", "alone", "apiece", "healthy", "active", "own",
    "scoreless", "outright", "straight", "apart", "aside", "overall",
    "decided", "decide", "decides", "settled", "separated", "outscored",
    "outrebounded", "trailing", "leading", "behind", "ahead", "within",
}
# Everything the pipeline already recognizes, reused as a single test. A token
# matching any of it is vocabulary, not a name.
KNOWN_VOCAB = "|".join((SUPPORTED_CONCEPT_WORDS, HOME_WORDS, AWAY_WORDS,
                        LOSS_WORDS, WIN_WORDS, FIRST_WORDS, LAST_WORDS))


def _is_question_word(token: str) -> bool:
    """Is this token part of the vocabulary questions are written in?

    :param token: one word from the question
    :return: True when the word names something other than a person or team
    """
    low = token.lower().strip(".,?!:;'’\"")
    if not low or low.isdigit():
        return True
    return (low in QUESTION_WORDS or low in PROPER_NOUN_STOPWORDS
            or low in SUBJECT_STOPWORDS or low in MONTHS
            or bool(re.search(KNOWN_VOCAB, low, re.IGNORECASE)))


def unrecognized_name_phrase(question_text: str, resolved: dict) -> str | None:
    """Find consecutive words that read like a name but matched nothing.

    The two existing guards each read one surface of the question: one its
    capitalization, the other the grammar slot after "did". A question can miss
    both at once -- "what is lebron james average 3 pointer" is lowercase and
    has no verb for the slot to hook onto -- and then no filter is built and the
    statistic is computed over the whole league.

    This reads the question's vocabulary instead. Every word a basketball
    question needs is already written down somewhere in this file, so a run of
    two or more consecutive words that are in none of it, and are not covered by
    a team or player that did resolve, is somebody's name that the data does not
    contain. Two words are required because a single unknown word is more often
    a synonym nobody listed than a person.

    :param question_text: the raw question
    :param resolved: output of resolve.resolve_question()
    :return: the unrecognized phrase, or None when everything is accounted for
    """
    runs = unrecognized_runs(question_text, resolved)
    longest = max(runs, key=len, default=[])
    return " ".join(longest) if len(longest) >= 2 else None


def unrecognized_runs(question_text: str, resolved: dict) -> list[list[str]]:
    """Group the question's unaccounted-for words into consecutive runs.

    Split out from unrecognized_name_phrase because how many words it takes to
    mean something depends on where the question already stands. On its own, one
    unknown word is weak evidence of a name. Inside a query that is computing
    somebody's statistic and has nobody to attribute it to, the same word is the
    missing subject, so audit_plan reads the runs directly.

    :param question_text: the raw question
    :param resolved: output of resolve.resolve_question()
    :return: runs of consecutive unrecognized words, in order
    """
    covered: set[str] = set()
    for team in resolved["teams"]:
        for part in (team["name"], team["city"], team["abbreviation"]):
            covered.update(w.lower() for w in re.split(r"[\s'’-]+", part or "") if w)
    for player in resolved["players"]:
        for part in (player["first_name"], player["last_name"]):
            covered.update(w.lower() for w in re.split(r"[\s'’-]+", part or "") if w)

    runs: list[list[str]] = []
    run: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z'’.-]*", question_text):
        # A possessive is the name, not a separate word: "jordan's" is checked
        # as "jordan" so the apostrophe cannot hide an invented player.
        bare = re.sub(r"['’]s$", "", token.lower().strip(".,?!:;'’\""))
        if len(bare) < 2 or bare in covered or _is_question_word(bare):
            if run:
                runs.append(run)
            run = []
            continue
        run.append(token)
    if run:
        runs.append(run)
    return runs


def audit_plan(plan: dict, question: dict, resolved: dict) -> list[str]:
    """List the things the question states that the plan does not constrain.

    This is the general form of every abstention rule in this pipeline. The
    failures worth abstaining on are not cases where the system was unsure --
    there is no confidence score to threshold, and a wrong answer here is a
    correct computation over a wrongly-constrained set. "How many games was
    Hollis Whitcombe injured for" returned 841 because the player filter had
    nowhere to live on a game-level query, so the count was of every game in the
    season, computed perfectly.

    What can be checked is coverage: each element the question asserts -- a
    named player, a named team, a date, a threshold, a role, an outcome -- must
    appear in the executed plan. Anything asserted and not used means the query
    answered a broader question than the one asked, and a broader question's
    answer is wrong however well it was computed.

    :param plan: the finalized plan
    :param question: one entry from questions/questions.json
    :param resolved: output of resolve.resolve_question()
    :return: human-readable descriptions of unused question elements
    """
    unused: list[str] = []
    game_filters, row_filters = plan["game_filters"], plan["row_filters"]

    # Which row filters a source actually applies. Box scores use all of them;
    # injury notes are filtered by player only; a game-level or recap query
    # reads no rows at all, so anything left there is inert -- which is how a
    # named player stopped narrowing anything.
    applied = APPLIED_ROW_FILTERS.get(plan["source"], set())
    row_scoped = bool(applied)

    named = [p for p in resolved["players"] if p["exact"]]
    if named:
        person_pinned = (row_scoped and (row_filters.get("person_id") is not None
                                         or row_filters.get("person_ids")))
        absent_pinned = game_filters.get("exclude_person_id") is not None
        note_scoped = plan["source"] == "injury_notes" and row_filters.get("person_id")
        if not (person_pinned or absent_pinned or note_scoped):
            unused.append(f"player {named[0]['full_name']}")

    if resolved["teams"]:
        team_pinned = any(game_filters.get(k) is not None
                          for k in ("team_id", "opponent_team_id")) or (
            row_scoped and any(row_filters.get(k) is not None
                               for k in ("team_id", "exclude_team_id")))
        if not team_pinned:
            unused.append(f"team {resolved['teams'][0]['name']}")

    # Anything the wording implies must survive into the plan. Re-deriving it
    # here and comparing is what catches a filter the source cannot hold.
    inferred_game, inferred_row, _ = infer_filters(
        question["question"], resolved, plan["source"])
    for key, value in inferred_game.items():
        if value is not None and game_filters.get(key) != value:
            unused.append(f"game filter {key}")
    for key, value in inferred_row.items():
        if value is None:
            continue
        if key not in applied:
            unused.append(f"row filter {key} (unused by {plan['source']})")
        elif row_filters.get(key) != value:
            unused.append(f"row filter {key}")

    # Coverage runs in the other direction too. A question that asks *who* and
    # produces nothing capable of naming a person has not answered it -- "who
    # was the most injured player" came back with a status and a reason and no
    # name at all, which reads as an answer without being one.
    if re.search(PERSON_QUESTION, question["question"], re.IGNORECASE):
        if not (NAMING_PRIMITIVES & set(plan["outputs"].values())):
            unused.append("question asks which person, plan names nobody")

    # And once more for questions that count. "How many home games did the
    # Sunhawks win" narrowed to the right 10 games and then reported "1164
    # points, across 10 games", because the shape it was handed asked for
    # points. The count was exact; leading with a statistic nobody asked about
    # is what made it read as an answer to some other question.
    if re.search(COUNT_QUESTION, question["question"], re.IGNORECASE):
        # A ranking whose inner combiner counts rows reports its count as the
        # aggregate, so "agg" is a count there too.
        ranked_count = (plan["aggregate"]["op"] in ("argmax", "argmin")
                        and plan["aggregate"]["inner"] == "count"
                        and "agg" in plan["outputs"].values())
        if not (COUNTING_PRIMITIVES & set(plan["outputs"].values()) or ranked_count):
            unused.append("question counts things, plan returns no count")

    # A per-player statistic belonging to nobody in particular. "What is lebron
    # james average 3 pointer" answered "26 fg3_made, across 841 games": the
    # name matched no one, so no filter was built, and the average ran over
    # every player in the league. The two guards for unrecognized names read
    # the question -- one its capitalization, one its grammar -- and this one
    # was lowercase with no verb for the grammar to hook onto, so both passed
    # it through.
    #
    # Reading the plan's scope instead catches it whatever the phrasing. A box
    # score total is somebody's: it is pinned to a player, ranked to name one,
    # or restricted to a team. Belonging to none of the three means the subject
    # of the question never made it into the query.
    if plan["source"] == "player_box_scores" and "agg" in plan["outputs"].values():
        pinned = (row_filters.get("person_id") is not None
                  or row_filters.get("person_ids"))
        names_someone = "group_name" in plan["outputs"].values()
        team_scoped = any(row_filters.get(k) is not None
                          for k in ("team_id", "exclude_team_id")) or any(
            game_filters.get(k) is not None
            for k in ("team_id", "opponent_team_id"))
        # Naming a real team is only enough when every other word in the
        # question is accounted for. "What is jordan's average points against
        # the Outriders" is team-scoped and still about nobody: the invented name
        # built no filter, so the average ran over every opponent the Outriders
        # played. One unrecognized word is enough to say so here, where the
        # query is already computing a person's statistic and has no person to
        # attribute it to.
        if team_scoped and unrecognized_runs(question["question"], resolved):
            team_scoped = False
        if not (pinned or names_someone or team_scoped):
            unused.append("statistic belongs to no player, team or ranking")

    # The same reasoning one level up, for the subject the question named and
    # the query never found. An invented team is harder to spot than an
    # invented player, because a season-wide question about games is often
    # legitimate -- "what was the biggest blowout" names nobody and is a fair
    # question. The unaccounted-for word is what separates them: "what is the
    # fakers average margin" came back answerable with nothing in it, having
    # quietly ranged over every game in the league.
    if unrecognized_runs(question["question"], resolved):
        narrowed = (row_filters.get("person_id") is not None
                    or row_filters.get("person_ids")
                    or any(row_filters.get(k) is not None
                           for k in ("team_id", "exclude_team_id"))
                    or any(game_filters.get(k) is not None
                           for k in ("team_id", "opponent_team_id")))
        if not narrowed:
            unused.append("names something unrecognized and narrowed to nobody")

    # A record belongs to a team. "Who had the best record this season" was
    # answered "Tavian Belcourt, 2143 points": the bare "who" sent it to a
    # person shape, and a point total came along in place of a win-loss record
    # nobody had computed. Its "which team" twin already abstains.
    if (re.search(TEAM_CONCEPT, question["question"], re.IGNORECASE)
            and plan["source"] == "player_box_scores"):
        unused.append("question asks about a team's record, plan reads box scores")

    # "Averaged 20 or more" sets a floor on the average, not on any single
    # game. The DSL has row thresholds and it has averages, but nothing that
    # puts a threshold *on* an average, so the filter landed per-game: "how
    # many players averaged 20 or more points" answered 312, the number who
    # ever scored 20 in a game, where 40 actually averaged it. The condition
    # cannot be represented, so the question is refused rather than answered
    # as the different question the query happens to express.
    if (re.search(AVERAGE_WORDS, question["question"], re.IGNORECASE)
            and any(row_filters.get(k) is not None
                    for k in ("min_points", "min_rebounds", "min_assists"))):
        unused.append("threshold applies to an average, which no filter expresses")

    # A threshold with no superlative asks for everyone who cleared it, not for
    # whoever cleared it by the most. The plan has only argmax to offer, so
    # "which Gannets bench player scored more than 10 points against the Outriders
    # in 2026" named Omari Moreau with 44 points -- one of the four players
    # who qualified, carrying a total nobody scored, since 44 is his 28 and 16
    # from two separate games added together.
    if ("group_name" in plan["outputs"].values()
            and any(row_filters.get(k) is not None
                    for k in ("min_points", "min_rebounds", "min_assists"))
            and not re.search(RANKING_WORDS, question["question"], re.IGNORECASE)):
        unused.append("question asks who cleared a threshold, plan ranks them instead")
    return unused


def finalize_outputs(plan: dict, question: dict) -> dict:
    """Settle the output mapping, with the field name winning over the model.

    Three rules, in order:

    1. A field name with a known meaning gets that meaning. "games" is a game
       count and "teams" is a team count, whatever the model mapped them to --
       it mapped both to the aggregate, so "how many teams did X play for"
       answered with his 82 games.
    2. A bare numeric field is the stat being aggregated, but only when the
       question actually names a stat this pipeline can compute. Otherwise
       there is nothing behind it: asked for a jersey number, the aggregate
       happily supplied one.
    3. Anything still unmapped is dropped, which makes the caller abstain.

    :param plan: the plan whose outputs are being settled, modified in place
    :param question: one entry from questions/questions.json
    :return: the same plan
    """
    spec = question["return"]
    probe = dict(plan, aggregate=dict(plan["aggregate"]))
    defaults = default_outputs(question, probe)
    declares_text = any(s.get("table") in ("game_recaps", "injury_notes")
                        for s in spec.get("evidence", []))
    nameable = infer_field(question["question"], plan["source"]) is not None

    final = {}
    for field, type_name in spec.items():
        if field in ("answerable", "evidence"):
            continue
        if field in FIELD_PRIMITIVES:
            final[field] = FIELD_PRIMITIVES[field]
        elif field in plan["outputs"]:
            final[field] = plan["outputs"][field]
        elif field in defaults:
            final[field] = defaults[field]
        else:
            continue
        # A numeric field standing in for "the aggregate" is only meaningful if
        # the question names something aggregatable.
        if (final[field] == "agg" and not declares_text
                and not nameable and field not in AGG_FIELDS):
            final.pop(field, None)

    if plan["source"] in ("game_recaps", "injury_notes"):
        allowed_text = text_primitives_for(plan["source"])
        for field, primitive in list(final.items()):
            if primitive not in allowed_text:
                final[field] = "text"

    for field, primitive in list(final.items()):
        allowed = (NUMERIC_PRIMITIVES if spec[field] in ("int", "float", "number")
                   else STRING_PRIMITIVES)
        if primitive not in allowed:
            final.pop(field, None)

    plan["outputs"] = final
    return plan


def _clean_outputs(raw, question: dict, source: str) -> dict:
    """Keep only output mappings that could actually produce a valid value.

    Three things are checked: the field is one the question declares, the
    primitive exists, and the primitive's type matches the declared type. A
    mapping that survives all three may still be wrong, but it cannot be
    nonsense -- and the caller fills any dropped field from the by-name defaults.

    For a prose source, a primitive the executor cannot compute is rewritten to
    "text" rather than dropped, because that is what it must have meant: the
    value is in the passage.

    :param raw: the model's `outputs` object, not trusted to be a dict
    :param question: one entry from questions/questions.json
    :param source: the plan's source table
    :return: a cleaned field -> primitive mapping
    """
    if not isinstance(raw, dict):
        return {}
    spec = question["return"]
    cleaned = {}
    for field, primitive in raw.items():
        if field in ("answerable", "evidence") or field not in spec:
            continue
        if primitive not in PRIMITIVES:
            continue
        if source in ("game_recaps", "injury_notes") and primitive not in text_primitives_for(source):
            primitive = "text"
        allowed = NUMERIC_PRIMITIVES if spec[field] == "int" else STRING_PRIMITIVES
        if primitive not in allowed:
            continue
        cleaned[field] = primitive
    return cleaned


def validate_plan(raw: dict, resolved: dict, question: dict) -> dict | None:
    """Coerce the model's plan into a safe, executable plan, or reject it.

    Enforces the closed-world rule on ids: every person_id and team_id in the
    plan must be one resolve.py actually found in the question. Without this a
    model that hallucinates a plausible id would produce a fully grounded,
    confidently wrong answer about a different player -- the worst possible
    failure mode, because nothing downstream could detect it.

    :param raw: parsed JSON from the model
    :param resolved: output of resolve.resolve_question()
    :param question: one entry from questions/questions.json
    :return: a validated plan, or None if it cannot be made safe
    """
    if not isinstance(raw, dict):
        return None

    plan = blank_plan()

    source = raw.get("source")
    if source in SOURCES:
        plan["source"] = source

    plan["game_filters"] = _clean_filters(raw.get("game_filters"), GAME_FILTER_KEYS)
    plan["row_filters"] = _clean_filters(raw.get("row_filters"), ROW_FILTER_KEYS)

    select = raw.get("game_select")
    if select in GAME_SELECTS:
        plan["game_select"] = select
    elif isinstance(select, int) and 1 <= select <= 200:
        plan["game_select"] = select
    else:
        plan["game_select"] = "all"

    agg = raw.get("aggregate") if isinstance(raw.get("aggregate"), dict) else {}
    op = agg.get("op")
    field = agg.get("field")
    inner = agg.get("inner")
    try:
        rank = max(1, min(MAX_RANK, int(agg.get("rank", 1))))
    except (TypeError, ValueError):
        rank = 1
    plan["aggregate"] = {
        "op": op if op in AGG_OPS else "none",
        "field": field if field in AGG_FIELDS else "points",
        "group_by": agg.get("group_by") if agg.get("group_by") in GROUP_KEYS else None,
        "inner": inner if inner in INNER_OPS else "sum",
        "rank": rank,
    }
    # argmax/argmin without a grouping key is meaningless; treat it as a plain
    # max/min over the rows so the plan still executes deterministically.
    if plan["aggregate"]["op"] in ("argmax", "argmin") and plan["aggregate"]["group_by"] is None:
        plan["aggregate"]["op"] = "max" if plan["aggregate"]["op"] == "argmax" else "min"

    plan["outputs"] = _clean_outputs(raw.get("outputs"), question, plan["source"])

    # Closed-world id check. Anything not traceable to resolution is dropped
    # rather than trusted.
    allowed_teams = {t["team_id"] for t in resolved["teams"]}
    allowed_people = {p["player_id"] for p in resolved["players"]}
    if plan["game_filters"].get("venue") not in VENUES:
        plan["game_filters"].pop("venue", None)
    if plan["game_filters"].get("opponent_conference") not in CONFERENCES:
        plan["game_filters"].pop("opponent_conference", None)
    if plan["row_filters"].get("position") not in POSITIONS:
        plan["row_filters"].pop("position", None)

    # A list of people is subject to the same closed-world rule as a single one.
    raw_ids = plan["row_filters"].get("person_ids")
    if raw_ids is not None:
        try:
            kept = [int(p) for p in raw_ids if int(p) in allowed_people]
        except (TypeError, ValueError):
            kept = []
        if kept:
            plan["row_filters"]["person_ids"] = kept
        else:
            plan["row_filters"].pop("person_ids", None)

    for scope, key, allowed in (
        ("game_filters", "team_id", allowed_teams),
        ("game_filters", "opponent_team_id", allowed_teams),
        ("game_filters", "exclude_person_id", allowed_people),
        ("row_filters", "team_id", allowed_teams),
        ("row_filters", "exclude_team_id", allowed_teams),
        ("row_filters", "person_id", allowed_people),
    ):
        value = plan[scope].get(key)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            plan[scope].pop(key, None)
            continue
        if value in allowed:
            plan[scope][key] = value
        else:
            plan[scope].pop(key, None)

    # An empty `outputs` is not fatal. The filters the model got right are worth
    # keeping, and the caller fills missing output mappings from the field names.
    return plan
