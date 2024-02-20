class NBATeamConverter:
    """
    A class to convert between various identifiers of NBA teams such as team ID, abbreviation,
    short name, full name, or any of the alternatives.
    """

    # Team data with details for each team. Include abbreviation, full name, short name, and alternatives.
    teams_data = {
        "1610612737": {
            "abbreviation": "ATL",
            "full_name": "Atlanta Hawks",
            "short_name": "Hawks",
            "alternatives": [
                "STL",
                "Tri-Cities Blackhawks",
                "MLH",
                "TRI",
                "atlanta-hawks",
                "Milwaukee Hawks",
                "St. Louis Hawks",
            ],
        },
        "1610612751": {
            "abbreviation": "BKN",
            "full_name": "Brooklyn Nets",
            "short_name": "Nets",
            "alternatives": [
                "NJA",
                "BK",
                "brooklyn-nets",
                "NYN",
                "NJN",
                "New York Nets",
                "BRK",
                "New Jersey Americans",
                "New Jersey Nets",
            ],
        },
        "1610612738": {
            "abbreviation": "BOS",
            "full_name": "Boston Celtics",
            "short_name": "Celtics",
            "alternatives": ["boston-celtics"],
        },
        "1610612766": {
            "abbreviation": "CHA",
            "full_name": "Charlotte Hornets",
            "short_name": "Hornets",
            "alternatives": [
                "charlotte-hornets",
                "Charlotte Bobcats",
                "CHH",
                "CHO",
            ],
        },
        "1610612739": {
            "abbreviation": "CLE",
            "full_name": "Cleveland Cavaliers",
            "short_name": "Cavaliers",
            "alternatives": ["cleveland-cavaliers"],
        },
        "1610612741": {
            "abbreviation": "CHI",
            "full_name": "Chicago Bulls",
            "short_name": "Bulls",
            "alternatives": ["chicago-bulls"],
        },
        "1610612742": {
            "abbreviation": "DAL",
            "full_name": "Dallas Mavericks",
            "short_name": "Mavericks",
            "alternatives": ["dallas-mavericks"],
        },
        "1610612743": {
            "abbreviation": "DEN",
            "full_name": "Denver Nuggets",
            "short_name": "Nuggets",
            "alternatives": ["Denver Rockets", "DNR", "denver-nuggets"],
        },
        "1610612765": {
            "abbreviation": "DET",
            "full_name": "Detroit Pistons",
            "short_name": "Pistons",
            "alternatives": ["FTW", "Fort Wayne Pistons", "detroit-pistons"],
        },
        "1610612744": {
            "abbreviation": "GSW",
            "full_name": "Golden State Warriors",
            "short_name": "Warriors",
            "alternatives": [
                "PHW",
                "GS",
                "Philadelphia Warriors",
                "San Francisco Warriors",
                "SFW",
                "golden-state-warriors",
            ],
        },
        "1610612745": {
            "abbreviation": "HOU",
            "full_name": "Houston Rockets",
            "short_name": "Rockets",
            "alternatives": ["SDR", "San Diego Rockets", "houston-rockets"],
        },
        "1610612754": {
            "abbreviation": "IND",
            "full_name": "Indiana Pacers",
            "short_name": "Pacers",
            "alternatives": ["indiana-pacers"],
        },
        "1610612746": {
            "abbreviation": "LAC",
            "full_name": "Los Angeles Clippers",
            "short_name": "Clippers",
            "alternatives": [
                "SDC",
                "los-angeles-clippers",
                "Buffalo Braves",
                "San Diego Clippers",
                "LA Clippers",
                "BUF",
            ],
        },
        "1610612747": {
            "abbreviation": "LAL",
            "full_name": "Los Angeles Lakers",
            "short_name": "Lakers",
            "alternatives": ["MNL", "los-angeles-lakers", "Minneapolis Lakers"],
        },
        "1610612763": {
            "abbreviation": "MEM",
            "full_name": "Memphis Grizzlies",
            "short_name": "Grizzlies",
            "alternatives": ["memphis-grizzlies", "VAN", "Vancouver Grizzlies"],
        },
        "1610612748": {
            "abbreviation": "MIA",
            "full_name": "Miami Heat",
            "short_name": "Heat",
            "alternatives": ["miami-heat"],
        },
        "1610612749": {
            "abbreviation": "MIL",
            "full_name": "Milwaukee Bucks",
            "short_name": "Bucks",
            "alternatives": ["milwaukee-bucks"],
        },
        "1610612750": {
            "abbreviation": "MIN",
            "full_name": "Minnesota Timberwolves",
            "short_name": "Timberwolves",
            "alternatives": ["minnesota-timberwolves"],
        },
        "1610612752": {
            "abbreviation": "NYK",
            "full_name": "New York Knicks",
            "short_name": "Knicks",
            "alternatives": ["new-york-knicks", "NY"],
        },
        "1610612740": {
            "abbreviation": "NOP",
            "full_name": "New Orleans Pelicans",
            "short_name": "Pelicans",
            "alternatives": [
                "new-orleans-pelicans",
                "NOH",
                "New Orleans Hornets",
                "NOK",
                "NO",
                "New Orleans/Oklahoma City Hornets",
            ],
        },
        "1610612760": {
            "abbreviation": "OKC",
            "full_name": "Oklahoma City Thunder",
            "short_name": "Thunder",
            "alternatives": ["Seattle SuperSonics", "oklahoma-city-thunder", "SEA"],
        },
        "1610612753": {
            "abbreviation": "ORL",
            "full_name": "Orlando Magic",
            "short_name": "Magic",
            "alternatives": ["orlando-magic"],
        },
        "1610612755": {
            "abbreviation": "PHI",
            "full_name": "Philadelphia 76ers",
            "short_name": "76ers",
            "alternatives": [
                "PHL",
                "Syracuse Nationals",
                "SYR",
                "philadelphia-76ers",
            ],
        },
        "1610612756": {
            "abbreviation": "PHX",
            "full_name": "Phoenix Suns",
            "short_name": "Suns",
            "alternatives": ["phoenix-suns", "PHO"],
        },
        "1610612757": {
            "abbreviation": "POR",
            "full_name": "Portland Trail Blazers",
            "short_name": "Trail Blazers",
            "alternatives": ["portland-trail-blazers"],
        },
        "1610612759": {
            "abbreviation": "SAS",
            "full_name": "San Antonio Spurs",
            "short_name": "Spurs",
            "alternatives": [
                "DLC",
                "SAN",
                "Dallas Chaparrals",
                "SA",
                "san-antonio-spurs",
            ],
        },
        "1610612758": {
            "abbreviation": "SAC",
            "full_name": "Sacramento Kings",
            "short_name": "Kings",
            "alternatives": [
                "KCO",
                "KCK",
                "Kansas City-Omaha Kings",
                "Kansas City Kings",
                "sacramento-kings",
                "Cincinnati Royals",
                "CIN",
                "Rochester Royals",
                "ROC",
            ],
        },
        "1610612761": {
            "abbreviation": "TOR",
            "full_name": "Toronto Raptors",
            "short_name": "Raptors",
            "alternatives": ["toronto-raptors"],
        },
        "1610612762": {
            "abbreviation": "UTA",
            "full_name": "Utah Jazz",
            "short_name": "Jazz",
            "alternatives": ["NOJ", "utah-jazz", "New Orleans Jazz", "UTAH"],
        },
        "1610612764": {
            "abbreviation": "WAS",
            "full_name": "Washington Wizards",
            "short_name": "Wizards",
            "alternatives": [
                "WSH",
                "Washington Bullets",
                "CAP",
                "BAL",
                "Baltimore Bullets",
                "CHP",
                "CHZ",
                "Chicago Packers",
                "WSB",
                "washington-wizards",
                "Capital Bullets",
                "Chicago Zephyrs",
            ],
        },
    }

    # Lookup dictionary for quick ID retrieval based on any identifier.
    lookup_dict = {
        "1610612737": "1610612737",
        "ATL": "1610612737",
        "Hawks": "1610612737",
        "Atlanta Hawks": "1610612737",
        "STL": "1610612737",
        "Tri-Cities Blackhawks": "1610612737",
        "MLH": "1610612737",
        "TRI": "1610612737",
        "atlanta-hawks": "1610612737",
        "Milwaukee Hawks": "1610612737",
        "St. Louis Hawks": "1610612737",
        "1610612751": "1610612751",
        "BKN": "1610612751",
        "Nets": "1610612751",
        "Brooklyn Nets": "1610612751",
        "NJA": "1610612751",
        "BK": "1610612751",
        "brooklyn-nets": "1610612751",
        "NYN": "1610612751",
        "NJN": "1610612751",
        "New York Nets": "1610612751",
        "BRK": "1610612751",
        "New Jersey Americans": "1610612751",
        "New Jersey Nets": "1610612751",
        "1610612738": "1610612738",
        "BOS": "1610612738",
        "Celtics": "1610612738",
        "Boston Celtics": "1610612738",
        "boston-celtics": "1610612738",
        "1610612766": "1610612766",
        "CHA": "1610612766",
        "Hornets": "1610612766",
        "Charlotte Hornets": "1610612766",
        "charlotte-hornets": "1610612766",
        "Charlotte Bobcats": "1610612766",
        "CHH": "1610612766",
        "CHO": "1610612766",
        "1610612739": "1610612739",
        "CLE": "1610612739",
        "Cavaliers": "1610612739",
        "Cleveland Cavaliers": "1610612739",
        "cleveland-cavaliers": "1610612739",
        "1610612741": "1610612741",
        "CHI": "1610612741",
        "Bulls": "1610612741",
        "Chicago Bulls": "1610612741",
        "chicago-bulls": "1610612741",
        "1610612742": "1610612742",
        "DAL": "1610612742",
        "Mavericks": "1610612742",
        "Dallas Mavericks": "1610612742",
        "dallas-mavericks": "1610612742",
        "1610612743": "1610612743",
        "DEN": "1610612743",
        "Nuggets": "1610612743",
        "Denver Nuggets": "1610612743",
        "Denver Rockets": "1610612743",
        "DNR": "1610612743",
        "denver-nuggets": "1610612743",
        "1610612765": "1610612765",
        "DET": "1610612765",
        "Pistons": "1610612765",
        "Detroit Pistons": "1610612765",
        "FTW": "1610612765",
        "Fort Wayne Pistons": "1610612765",
        "detroit-pistons": "1610612765",
        "1610612744": "1610612744",
        "GSW": "1610612744",
        "Warriors": "1610612744",
        "Golden State Warriors": "1610612744",
        "PHW": "1610612744",
        "GS": "1610612744",
        "Philadelphia Warriors": "1610612744",
        "San Francisco Warriors": "1610612744",
        "SFW": "1610612744",
        "golden-state-warriors": "1610612744",
        "1610612745": "1610612745",
        "HOU": "1610612745",
        "Rockets": "1610612745",
        "Houston Rockets": "1610612745",
        "SDR": "1610612745",
        "San Diego Rockets": "1610612745",
        "houston-rockets": "1610612745",
        "1610612754": "1610612754",
        "IND": "1610612754",
        "Pacers": "1610612754",
        "Indiana Pacers": "1610612754",
        "indiana-pacers": "1610612754",
        "1610612746": "1610612746",
        "LAC": "1610612746",
        "Clippers": "1610612746",
        "Los Angeles Clippers": "1610612746",
        "SDC": "1610612746",
        "los-angeles-clippers": "1610612746",
        "Buffalo Braves": "1610612746",
        "San Diego Clippers": "1610612746",
        "LA Clippers": "1610612746",
        "BUF": "1610612746",
        "1610612747": "1610612747",
        "LAL": "1610612747",
        "Lakers": "1610612747",
        "Los Angeles Lakers": "1610612747",
        "MNL": "1610612747",
        "los-angeles-lakers": "1610612747",
        "Minneapolis Lakers": "1610612747",
        "1610612763": "1610612763",
        "MEM": "1610612763",
        "Grizzlies": "1610612763",
        "Memphis Grizzlies": "1610612763",
        "memphis-grizzlies": "1610612763",
        "VAN": "1610612763",
        "Vancouver Grizzlies": "1610612763",
        "1610612748": "1610612748",
        "MIA": "1610612748",
        "Heat": "1610612748",
        "Miami Heat": "1610612748",
        "miami-heat": "1610612748",
        "1610612749": "1610612749",
        "MIL": "1610612749",
        "Bucks": "1610612749",
        "Milwaukee Bucks": "1610612749",
        "milwaukee-bucks": "1610612749",
        "1610612750": "1610612750",
        "MIN": "1610612750",
        "Timberwolves": "1610612750",
        "Minnesota Timberwolves": "1610612750",
        "minnesota-timberwolves": "1610612750",
        "1610612752": "1610612752",
        "NYK": "1610612752",
        "Knicks": "1610612752",
        "New York Knicks": "1610612752",
        "new-york-knicks": "1610612752",
        "NY": "1610612752",
        "1610612740": "1610612740",
        "NOP": "1610612740",
        "Pelicans": "1610612740",
        "New Orleans Pelicans": "1610612740",
        "new-orleans-pelicans": "1610612740",
        "NOH": "1610612740",
        "New Orleans Hornets": "1610612740",
        "NOK": "1610612740",
        "NO": "1610612740",
        "New Orleans/Oklahoma City Hornets": "1610612740",
        "1610612760": "1610612760",
        "OKC": "1610612760",
        "Thunder": "1610612760",
        "Oklahoma City Thunder": "1610612760",
        "Seattle SuperSonics": "1610612760",
        "oklahoma-city-thunder": "1610612760",
        "SEA": "1610612760",
        "1610612753": "1610612753",
        "ORL": "1610612753",
        "Magic": "1610612753",
        "Orlando Magic": "1610612753",
        "orlando-magic": "1610612753",
        "1610612755": "1610612755",
        "PHI": "1610612755",
        "76ers": "1610612755",
        "Philadelphia 76ers": "1610612755",
        "PHL": "1610612755",
        "Syracuse Nationals": "1610612755",
        "SYR": "1610612755",
        "philadelphia-76ers": "1610612755",
        "1610612756": "1610612756",
        "PHX": "1610612756",
        "Suns": "1610612756",
        "Phoenix Suns": "1610612756",
        "phoenix-suns": "1610612756",
        "PHO": "1610612756",
        "1610612757": "1610612757",
        "POR": "1610612757",
        "Trail Blazers": "1610612757",
        "Portland Trail Blazers": "1610612757",
        "portland-trail-blazers": "1610612757",
        "1610612759": "1610612759",
        "SAS": "1610612759",
        "Spurs": "1610612759",
        "San Antonio Spurs": "1610612759",
        "DLC": "1610612759",
        "SAN": "1610612759",
        "Dallas Chaparrals": "1610612759",
        "SA": "1610612759",
        "san-antonio-spurs": "1610612759",
        "1610612758": "1610612758",
        "SAC": "1610612758",
        "Kings": "1610612758",
        "Sacramento Kings": "1610612758",
        "KCO": "1610612758",
        "KCK": "1610612758",
        "Kansas City-Omaha Kings": "1610612758",
        "Kansas City Kings": "1610612758",
        "sacramento-kings": "1610612758",
        "Cincinnati Royals": "1610612758",
        "CIN": "1610612758",
        "Rochester Royals": "1610612758",
        "ROC": "1610612758",
        "1610612761": "1610612761",
        "TOR": "1610612761",
        "Raptors": "1610612761",
        "Toronto Raptors": "1610612761",
        "toronto-raptors": "1610612761",
        "1610612762": "1610612762",
        "UTA": "1610612762",
        "Jazz": "1610612762",
        "Utah Jazz": "1610612762",
        "NOJ": "1610612762",
        "utah-jazz": "1610612762",
        "New Orleans Jazz": "1610612762",
        "UTAH": "1610612762",
        "1610612764": "1610612764",
        "WAS": "1610612764",
        "Wizards": "1610612764",
        "Washington Wizards": "1610612764",
        "WSH": "1610612764",
        "Washington Bullets": "1610612764",
        "CAP": "1610612764",
        "BAL": "1610612764",
        "Baltimore Bullets": "1610612764",
        "CHP": "1610612764",
        "CHZ": "1610612764",
        "Chicago Packers": "1610612764",
        "WSB": "1610612764",
        "washington-wizards": "1610612764",
        "Capital Bullets": "1610612764",
        "Chicago Zephyrs": "1610612764",
    }

    @classmethod
    def get_team_id(cls, identifier):
        """
        Retrieve the team ID based on any given identifier.

        Args:
            identifier (str|int): The identifier to lookup (team ID, abbreviation, short name, full name, or alternatives).

        Returns:
            int|str: The corresponding team ID or a message indicating an unknown identifier.
        """
        return cls.lookup_dict.get(identifier, "Unknown ID")

    @classmethod
    def get_abbreviation(cls, identifier):
        """
        Retrieve the team abbreviation based on any given identifier.

        Args:
            identifier (str|int): The identifier to lookup.

        Returns:
            str: The corresponding team abbreviation or a message indicating an unknown identifier.
        """
        team_id = cls.get_team_id(identifier)
        return (
            cls.teams_data[team_id]["abbreviation"]
            if team_id != "Unknown ID"
            else "Unknown Abbreviation"
        )

    @classmethod
    def get_short_name(cls, identifier):
        """
        Retrieve the team short name based on any given identifier.

        Args:
            identifier (str|int): The identifier to lookup.

        Returns:
            str: The corresponding team short name or a message indicating an unknown identifier.
        """
        team_id = cls.get_team_id(identifier)
        return (
            cls.teams_data[team_id]["short_name"]
            if team_id != "Unknown ID"
            else "Unknown Short Name"
        )

    @classmethod
    def get_full_name(cls, identifier):
        """
        Retrieve the team full name based on any given identifier.

        Args:
            identifier (str|int): The identifier to lookup.

        Returns:
            str: The corresponding team full name or a message indicating an unknown identifier.
        """
        team_id = cls.get_team_id(identifier)
        return (
            cls.teams_data[team_id]["full_name"]
            if team_id != "Unknown ID"
            else "Unknown Full Name"
        )
