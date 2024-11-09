from enum import Enum

REGION_LABEL = {
    'Central Region': {'residential': 0.3, 'commercial': 0.5, 'industrial': 0.1, 'others': 0.1},
    'East Region': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1},
    'North East Region': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1},
    'North Region': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1},
    'West Region': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1},
    'Bishan': {'residential': 0.6, 'commercial': 0.2, 'industrial': 0.2, 'others': 0.0},
    'Bukit Merah': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Bukit Timah': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Downtown Core': {'residential': 0.1, 'commercial': 0.8, 'industrial': 0.0, 'others': 0.1},
    'Geylang': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Kallang': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.1, 'others': 0.0},
    'Marine Parade': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Museum': {'residential': 0.1, 'commercial': 0.8, 'industrial': 0.0, 'others': 0.1},
    'Newton': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Novena': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Orchard': {'residential': 0.2, 'commercial': 0.7, 'industrial': 0.0, 'others': 0.1},
    'Outram': {'residential': 0.4, 'commercial': 0.5, 'industrial': 0.0, 'others': 0.1},
    'Queenstown': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'River Valley': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1},
    'Rochor': {'residential': 0.4, 'commercial': 0.5, 'industrial': 0.0, 'others': 0.1},
    'Singapore River': {'residential': 0.1, 'commercial': 0.8, 'industrial': 0.0, 'others': 0.1},
    'Southern Islands': {'residential': 0.0, 'commercial': 0.1, 'industrial': 0.0, 'others': 0.9},
    'Tanglin': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Toa Payoh': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Bedok': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Changi': {'residential': 0.0, 'commercial': 0.2, 'industrial': 0.7, 'others': 0.1},
    'Pasir Ris': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Paya Lebar': {'residential': 0.5, 'commercial': 0.4, 'industrial': 0.0, 'others': 0.1},
    'Tampines': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Ang Mo Kio': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Hougang': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Punggol': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Seletar': {'residential': 0.0, 'commercial': 0.1, 'industrial': 0.8, 'others': 0.1},
    'Sengkang': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Serangoon': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Mandai': {'residential': 0.0, 'commercial': 0.0, 'industrial': 0.9, 'others': 0.1},
    'Sembawang': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Sungei Kadut': {'residential': 0.0, 'commercial': 0.0, 'industrial': 0.9, 'others': 0.1},
    'Woodlands': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Yishun': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.0, 'others': 0.1},
    'Bukit Batok': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Bukit Panjang': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Choa Chu Kang': {'residential': 0.8, 'commercial': 0.1, 'industrial': 0.1, 'others': 0.0},
    'Clementi': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Jurong East': {'residential': 0.4, 'commercial': 0.5, 'industrial': 0.0, 'others': 0.1},
    'Jurong West': {'residential': 0.7, 'commercial': 0.2, 'industrial': 0.1, 'others': 0.0},
    'Pioneer': {'residential': 0.0, 'commercial': 0.1, 'industrial': 0.8, 'others': 0.1},
    'Tengah': {'residential': 0.6, 'commercial': 0.3, 'industrial': 0.0, 'others': 0.1}
}


REGION_CODE = {
    "residential": 0,
    "commercial": 1,
    "industrial": 2,
    "others": 3,
}

REGION_CODE_INVERSE = {
    0: "residential",
    1: "commercial",
    2: "industrial",
    3: "others",
}

POLICY_MAP = {
    0: "Keep supply as this",
    1: "Increase supply",
    2: "Decrease supply",
}

class State(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
