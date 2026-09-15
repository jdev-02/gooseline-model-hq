"""Team codes to the names a person actually says. "CWS @ CLE" is data; "White
Sox at Guardians" is a sentence. Codes are the ones StatsAPI and nflverse
use, which are also Kalshi's."""

MLB = {
    "AZ": "Diamondbacks", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds", "CLE": "Guardians",
    "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
    "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "ATH": "Athletics",
    "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SF": "Giants",
    "SEA": "Mariners", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WSH": "Nationals",
}

NFL = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LV": "Raiders", "LAC": "Chargers", "LA": "Rams", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SF": "49ers",
    "SEA": "Seahawks", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}


def mlb(code):
    return MLB.get(str(code), str(code))


def nfl(code):
    return NFL.get(str(code), str(code))
