# Mapping from speaker title variants seen in data to their canonical form
CANONICAL_SPEAKER_TITLES = {
    "u.s. house of representative": "u.s. house of representatives",
    "u.s. representativej": "u.s. representative",
    "talks show host": "talk show host",
    "u. s. congressman": "u.s. congressman",
    "politican action committee": "political action committee",
    "retired": "retiree",
    "restauranteur": "restaurateur"
}

SIX_WAY_LABEL_TO_BINARY = {
    "pants-fire": False,
    "barely-true": False,
    "false": False,
    "true": True,
    "half-true": True,
    "mostly-true": True
}

CANONICAL_STATE = {
    "tennesse": "tennessee",
    "district of columbia": "washington d.c.",
    "washington dc": "washington d.c.",
    "washington, d.c.": "washington d.c.",
    "washington d.c.": "washington d.c.",
    "tex": "texas",
    "texas": "texas",
    "washington state": "washington",
    "washington": "washington",
    "virgina": "virginia",
    "virgiia": "virginia",
    "virginia": "virginia",
    "pennsylvania": "pennsylvania",
    "pa - pennsylvania": "pennsylvania",
    "rhode island": "rhode island",
    "rhode island": "rhode island",
    "ohio": "ohio",
    "ohio": "ohio"
}

PARTY_AFFILIATIONS = {
    "republican", "democrat", "none", "organization", "independent",
    "columnist", "activist", "talk-show-host", "libertarian",
    "newsmaker", "journalist", "labor-leader", "state-official",
    "business-leader", "education-official", "tea-party-member",
    "green", "liberal-party-canada", "government-body", "Moderate",
    "democratic-farmer-labor", "ocean-state-tea-party-action",
    "constitution-party"
}
