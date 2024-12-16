import os
from functools import lru_cache
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Optional, List, Tuple, Type, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import json

import numpy
import pandas as pd
import pycountry
import numpy as np

"""
GDELT HEAD:
Year,Actor1Country,Actor2Country,weighted_sum_avgtone,weighted_sum_goldstein,sum_nummentions
1979,ABW,NLD,5,3.4,4
1979,AFG,AFG,5.5960831546356324,0.50163652024117145,1161
1979,AFG,ARE,6.79611650485437,1.9,8
1979,AFG,BEL,4.96613995485327,2.5,18
1979,AFG,BGR,6.0363729323650475,3.0799999999999992,100
1979,AFG,CAN,2.4390243902439,-10,18
1979,AFG,CHN,5.8165367352912618,-1.2832000000000003,125
1979,AFG,CUB,5.1874295187414967,3.4255813953488374,43
1979,AFG,CZE,9.1038284625995161,2.7956521739130431,46
"""

"""
MIGRATION HEAD:
head -n 6 migration/data/migration_imputed_RIKS_dec2021.csv
iso_or,origin,iso_des,destination,year,stock,flow,inflow,outflow
"AAB","igua and Barbuda","ABW","Aruba",1960,16,,,
"AAB","igua and Barbuda","ABW","Aruba",1961,16,0,,
"AAB","igua and Barbuda","ABW","Aruba",1962,15,-1,,
"AAB","igua and Barbuda","ABW","Aruba",1963,15,0,,
"AAB","igua and Barbuda","ABW","Aruba",1964,15,0,,
"""

"""
REFUGEE HEAD:
Year,"Country of origin","Country of asylum","Refugees under UNHCR's mandate",Asylum-seekers,"Returned refugees","IDPs of concern to UNHCR","Returned IDPss","Stateless persons","Others of concern","Other people in need of international protection","Host Community"
1960,"Unknown ",Austria,39000,0,0,0,0,0,0,,0
1960,"Unknown ",Belgium,55000,0,0,0,0,0,0,,0
1960,"Unknown ",Canada,48629,0,0,0,0,0,0,,0
1960,Angola,"Dem. Rep. of the Congo",150000,0,0,0,0,0,0,,0
1960,"Unknown ",Denmark,2300,0,0,0,0,0,0,,0
"""

CODE_MAPPING = {1: 'MAKE PUBLIC STATEMENT', 10: 'DEMAND', 11: 'DISAPPROVE', 12: 'REJECT', 13: 'THREATEN', 14: 'PROTEST', 15: 'EXHIBIT FORCE POSTURE', 16: 'REDUCE RELATIONS', 17: 'COERCE', 18: 'ASSAULT', 19: 'FIGHT', 2: 'APPEAL', 20: 'USE UNCONVENTIONAL MASS VIOLENCE', 21: 'Appeal for material cooperation, not specified below', 211: 'Appeal for economic cooperation', 212: 'Appeal for military cooperation', 213: 'Appeal for judicial cooperation', 214: 'Appeal for intelligence', 22: 'Appeal for diplomatic cooperation, such as policy support', 23: 'Appeal for aid, not specified below', 231: 'Appeal for economic aid', 232: 'Appeal for military aid', 233: 'Appeal for humanitarian aid', 234: 'Appeal for military protection or peacekeeping', 24: 'Appeal for political reform, not specified below', 241: 'Appeal for change in leadership', 242: 'Appeal for policy change', 243: 'Appeal for rights', 244: 'Appeal for change in institutions, regime', 25: 'Appeal to yield', 251: 'Appeal for easing of administrative sanctions', 252: 'Appeal for easing of popular dissent', 253: 'Appeal for release of persons or property', 254: 'Appeal for easing of economic sanctions, boycott, or embargo', 255: 'Appeal for target to allow international involvement (non-mediation)', 256: 'Appeal for de-escalation of military engagement', 26: 'Appeal to others to meet or negotiate', 27: 'Appeal to others to settle dispute', 28: 'Appeal to others to engage in or accept mediation', 3: 'EXPRESS INTENT TO COOPERATE', 30: 'Express intent to cooperate, not specified below', 31: 'Express intent to engage in material cooperation,  not specified below', 311: 'Express intent to cooperate economically', 312: 'Express intent to cooperate militarily', 313: 'Express intent to cooperate on judicial matters', 314: 'Express intent to cooperate on intelligence', 32: 'Express intent to provide diplomatic cooperation such as policy support', 33: 'Express intent to provide matyerial aid, not specified below', 331: 'Express intent to provide economic aid', 332: 'Express intent to provide military aid', 333: 'Express intent to provide humanitarian aid', 334: 'Express intent to provide military protection or peacekeeping', 34: 'Express intent to institute political reform, not specified below', 341: 'Express intent to change leadership', 342: 'Express intent to change policy', 343: 'Express intent to provide rights', 344: 'Express intent to change institutions, regime', 35: 'Express intent to yield, not specified below', 351: 'Express intent to ease administrative sanctions', 352: 'Express intent to ease popular dissent', 353: 'Express intent to release persons or property', 354: 'Express intent to ease economic sanctions, boycott, or embargo', 355: 'Express intent allow international involvement (not mediation)', 356: 'Express intent to de-escalate military engagement', 36: 'Express intent to meet or negotiate', 37: 'Express intent to settle dispute', 38: 'Express intent to accept mediation', 39: 'Express intent to mediate', 4: 'CONSULT', 40: 'Consult, not specified below', 41: 'Discuss by telephone', 42: 'Make a visit', 43: 'Host a visit', 44: 'Meet at a Ã’hirdÃ“location', 45: 'Mediate', 46: 'Engage in negotiation', 5: 'ENGAGE IN DIPLOMATIC COOPERATION', 50: 'Engage in diplomatic cooperation, not specified below', 51: 'Praise or endorse', 52: 'Defend verbally', 53: 'Rally support on behalf of', 54: 'Grant diplomatic recognition', 55: 'Apologize', 56: 'Forgive', 57: 'Sign formal agreement', 6: 'ENGAGE IN MATERIAL COOPERATION', 60: 'Engage in material cooperation, not specified below', 61: 'Cooperate economically', 62: 'Cooperate militarily', 63: 'Engage in judicial cooperation', 64: 'Share intelligence or information', 7: 'PROVIDE AID', 70: 'Provide aid, not specified below', 71: 'Provide economic aid', 72: 'Provide military aid', 73: 'Provide humanitarian aid', 74: 'Provide military protection or peacekeeping', 75: 'Grant asylum', 8: 'YIELD', 80: 'Yield, not specified below', 81: 'Ease administrative sanctions, not specified below', 811: 'Ease restrictions on political freedoms', 812: 'Ease ban on political parties or politicians', 813: 'Ease curfew', 814: 'Ease state of emergency or martial law', 82: 'Ease political dissent', 83: 'Accede to requests or demands for political reform not specified below', 831: 'Accede to demands for change in leadership', 832: 'Accede to demands for change in policy', 833: 'Accede to demands for rights', 834: 'Accede to demands for change in institutions, regime', 84: 'Return, release, not specified below', 841: 'Return, release person(s)', 842: 'Return, release property', 85: 'Ease economic sanctions, boycott, embargo', 86: 'Allow international involvement not specified below', 861: 'Receive deployment of peacekeepers', 862: 'Receive inspectors', 863: 'Allow delivery of humanitarian aid', 87: 'De-escalate military engagement', 871: 'Declare truce, ceasefire', 872: 'Ease military blockade', 873: 'Demobilize armed forces', 874: 'Retreat or surrender militarily', 9: 'INVESTIGATE', 90: 'Investigate, not specified below', 91: 'Investigate crime, corruption', 92: 'Investigate human rights abuses', 93: 'Investigate military action', 94: 'Investigate war crimes', 100: 'Demand, not specified below', 101: 'Demand information, investigation', 1011: 'Demand economic cooperation', 1012: 'Demand military cooperation', 1013: 'Demand judicial cooperation', 1014: 'Demand intelligence cooperation', 102: 'Demand policy support', 103: 'Demand aid, protection, or peacekeeping', 1031: 'Demand economic aid', 1032: 'Demand military aid', 1033: 'Demand humanitarian aid', 1034: 'Demand military protection or peacekeeping', 104: 'Demand political reform, not specified below', 1041: 'Demand change in leadership', 1042: 'Demand policy change', 1043: 'Demand rights', 1044: 'Demand change in institutions, regime', 105: 'Demand mediation', 1051: 'Demand easing of administrative sanctions', 1052: 'Demand easing of political dissent', 1053: 'Demand release of persons or property', 1054: 'Demand easing of economic sanctions, boycott, or embargo', 1055: 'Demand that target allows international involvement (non-mediation)', 1056: 'Demand de-escalation of military engagement', 106: 'Demand withdrawal', 107: 'Demand ceasefire', 108: 'Demand meeting, negotiation', 110: 'Disapprove, not specified below', 111: 'Criticize or denounce', 112: 'Accuse, not specified below', 1121: 'Accuse of crime, corruption', 1122: 'Accuse of human rights abuses', 1123: 'Accuse of aggression', 1124: 'Accuse of war crimes', 1125: 'Accuse of espionage, treason', 113: 'Rally opposition against', 114: 'Complain officially', 115: 'Bring lawsuit against', 116: 'Find guilty or liable (legally)', 120: 'Reject, not specified below', 121: 'Reject material cooperation', 1211: 'Reject economic cooperation', 1212: 'Reject military cooperation', 122: 'Reject request or demand for material aid, not specified below', 1221: 'Reject request for economic aid', 1222: 'Reject request for military aid', 1223: 'Reject request for humanitarian aid', 1224: 'Reject request for military protection or peacekeeping', 123: 'Reject request or demand for political reform, not specified below', 1231: 'Reject request for change in leadership', 1232: 'Reject request for policy change', 1233: 'Reject request for rights', 1234: 'Reject request for change in institutions, regime', 124: 'Refuse to yield, not specified below', 1241: 'Refuse to ease administrative sanctions', 1242: 'Refuse to ease popular dissent', 1243: 'Refuse to release persons or property', 1244: 'Refuse to ease economic sanctions, boycott, or embargo', 1245: 'Refuse to allow international involvement (non mediation)', 1246: 'Refuse to de-escalate military engagement', 125: 'Reject proposal to meet, discuss, or negotiate', 126: 'Reject mediation', 127: 'Reject plan, agreement to settle dispute', 128: 'Defy norms, law', 129: 'Veto', 130: 'Threaten, not specified below', 131: 'Threaten non-force, not specified below', 1311: 'Threaten to reduce or stop aid', 1312: 'Threaten to boycott, embargo, or sanction', 1313: 'Threaten to reduce or break relations', 132: 'Threaten with administrative sanctions, not specified below', 1321: 'Threaten to impose restrictions on political freedoms', 1322: 'Threaten to ban political parties or politicians', 1323: 'Threaten to impose curfew', 1324: 'Threaten to impose state of emergency or martial law', 133: 'Threaten political dissent, protest', 134: 'Threaten to halt negotiations', 135: 'Threaten to halt mediation', 136: 'Threaten to halt international involvement (non-mediation)', 137: 'Threaten with violent repression', 138: 'Threaten to use military force, not specified below', 1381: 'Threaten blockade', 1382: 'Threaten occupation', 1383: 'Threaten unconventional violence', 1384: 'Threaten conventional attack', 1385: 'Threaten attack with WMD', 139: 'Give ultimatum', 140: 'Engage in political dissent, not specified below', 141: 'Demonstrate or rally', 1411: 'Demonstrate for leadership change', 1412: 'Demonstrate for policy change', 1413: 'Demonstrate for rights', 1414: 'Demonstrate for change in institutions, regime', 142: 'Conduct hunger strike, not specified below', 1421: 'Conduct hunger strike for leadership change', 1422: 'Conduct hunger strike for policy change', 1423: 'Conduct hunger strike for rights', 1424: 'Conduct hunger strike for change in institutions, regime', 143: 'Conduct strike or boycott, not specified below', 1431: 'Conduct strike or boycott for leadership change', 1432: 'Conduct strike or boycott for policy change', 1433: 'Conduct strike or boycott for rights', 1434: 'Conduct strike or boycott for change in institutions, regime', 144: 'Obstruct passage, block', 1441: 'Obstruct passage to demand leadership change', 1442: 'Obstruct passage to demand policy change', 1443: 'Obstruct passage to demand rights', 1444: 'Obstruct passage to demand change in institutions, regime', 145: 'Protest violently, riot', 1451: 'Engage in violent protest for leadership change', 1452: 'Engage in violent protest for policy change', 1453: 'Engage in violent protest for rights', 1454: 'Engage in violent protest for  change in institutions, regime', 150: 'Demonstrate military or police power, not specified below', 151: 'Increase police alert status', 152: 'Increase military alert status', 153: 'Mobilize or increase police power', 154: 'Mobilize or increase armed forces', 160: 'Reduce relations, not specified below', 161: 'Reduce or break diplomatic relations', 162: 'Reduce or stop aid, not specified below', 1621: 'Reduce or stop economic assistance', 1622: 'Reduce or stop military assistance', 1623: 'Reduce or stop humanitarian assistance', 163: 'Impose embargo, boycott, or sanctions', 164: 'Halt negotiations', 165: 'Halt mediation', 166: 'Expel or withdraw, not specified below', 1661: 'Expel or withdraw peacekeepers', 1662: 'Expel or withdraw inspectors, observers', 1663: 'Expel or withdraw aid agencies', 170: 'Coerce, not specified below', 171: 'Seize or damage property, not specified below', 1711: 'Confiscate property', 1712: 'Destroy property', 172: 'Impose administrative sanctions, not specified below', 1721: 'Impose restrictions on political freedoms', 1722: 'Ban political parties or politicians', 1723: 'Impose curfew', 1724: 'Impose state of emergency or martial law', 173: 'Arrest, detain, or charge with legal action', 174: 'Expel or deport individuals', 175: 'Use tactics of violent repression', 180: 'Use unconventional violence, not specified below', 181: 'Abduct, hijack, or take hostage', 182: 'Physically assault, not specified below', 1821: 'Sexually assault', 1822: 'Torture', 1823: 'Kill by physical assault', 183: 'Conduct suicide, car, or other non-military bombing, not spec below', 1831: 'Carry out suicide bombing', 1832: 'Carry out car bombing', 1833: 'Carry out roadside bombing', 184: 'Use as human shield', 185: 'Attempt to assassinate', 186: 'Assassinate', 190: 'Use conventional military force, not specified below', 191: 'Impose blockade, restrict movement', 192: 'Occupy territory', 193: 'Fight with small arms and light weapons', 194: 'Fight with artillery and tanks', 195: 'Employ aerial weapons', 196: 'Violate ceasefire', 200: 'Use unconventional mass violence, not specified below', 201: 'Engage in mass expulsion', 202: 'Engage in mass killings', 203: 'Engage in ethnic cleansing', 204: 'Use weapons of mass destruction, not specified below', 2041: 'Use chemical, biological, or radiologicalweapons', 2042: 'Detonate nuclear weapons'}

COLOR_MAP = {
    "migration_in": "green",
    "migration_out": "red",
    "refugee": "purple",

}

def get_abs(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path

def load_df(path: str) -> pd.DataFrame:
    path = get_abs(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}. Please check the path and try again.')

    return pd.read_csv(path)

def lower_column_names(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.lower()

def url_valid(url: Any) -> bool:
    if not url or pd.isna(url) or not isinstance(url, str) or not url.startswith('http'):
        return False
    return True

country_iso_special_cases = {
    'Tibetan': 'CHN',
    'Dem. Rep. of the Congo': 'COD',
    'Syrian Arab Rep.': 'SYR',
    'Serbia and Kosovo: S/RES/1244 (1999)': 'SRB',  # Note: This is complicated, might w separate handling for Kosovo
    'Bolivia (Plurinational State of)': 'BOL',
    'United Rep. of Tanzania': 'TZA',
    "Lao People's Dem. Rep.": 'LAO',
    'Palestinian': 'PSE',  # Palestine, State of
    'Dominican Rep.': 'DOM',
    'Iran (Islamic Rep. of)': 'IRN',
    'Central African Rep.': 'CAF',
    'Rep. of Korea': 'KOR',  # South Korea
    "Cote d'Ivoire": 'CIV',
    'Venezuela (Bolivarian Republic of)': 'VEN',
    'Rep. of Moldova': 'MDA',
    'China, Hong Kong SAR': 'HKG',
    'Netherlands (Kingdom of the)': 'NLD',
    "Dem. People's Rep. of Korea": 'PRK',  # North Korea
    'China, Macao SAR': 'MAC',
    'Micronesia (Federated States of)': 'FSM',
    'Holy See': 'VAT',
    'Curacao': 'CUW',
    
    'Bahamas; The': 'BHS',
    'Bonaire; Sint Eustatius and Saba': 'BES',
    'Congo; Dem. Rep.': 'COD',
    'Congo; Rep.': 'COG',
    'Czech Rep.': 'CZE',
    'East and West Pakistan': 'PAK', 
    'Egypt; Arab Rep.': 'EGY',
    'Gambia; The': 'GMB',
    'Hong Kong; SAR China': 'HKG',
    'Iran; Islamic Rep.': 'IRN',
    "Korea; Dem. People's Rep.": 'PRK',
    'Korea; Rep.': 'KOR',
    'Kyrgyz Rep.': 'KGZ',
    'Macao SAR; China': 'MAC',
    'Micronesia; Fed. Sts.': 'FSM',
    'Netherlands; The': 'NLD',
    'Pitcairn Islands': 'PCN',
    'Sahrawi Arab Dem. Rep.': 'ESH', # western sahara
    'Slovak Rep.': 'SVK',
    'St. Helena': 'SHN',
    'St. Kitts and Nevis': 'KNA',
    'St. Lucia': 'LCA',
    'St. Pierre and Miquelon': 'SPM',
    'St. Vincent and the Grenadines': 'VCT',
    'Swaziland': 'SWZ', # eswatini
    'Turkey': 'TUR',
    'Vietnam; Dem. Rep.': 'VNM', 
    'Virgin Islands (U.S.)': 'VIR',
    'West Bank and Gaza': 'PSE', # palestine
    'Yemen; Rep.': 'YEM',

    # Countries that no longer exist and iso codes for them are deleted.
    # 'Czechoslovakia': 'CSK',
    # 'Netherlands Antilles': 'ANT',
    # 'Serbia-Montenegro': 'SCG',
    # 'USSR Soviet Union': 'SUN',
    # 'Yemen Arab Rep.': 'YEM', 
    # 'Yugoslavia': 'YUG',
}
country_iso_special_cases = {key: pycountry.countries.lookup(value) for key, value in country_iso_special_cases.items()}

def get_country_iso3(country: str, fuzzy: bool = False, throw: bool = True) -> Optional[str]:
    country = get_country(country, fuzzy, throw)
    if country:
        return country.alpha_3
    return None

@lru_cache(maxsize=1024)
def get_country(country:str, fuzzy: bool = False, throw: bool = True) -> Optional[Any]:
    if not country or pd.isna(country) or not isinstance(country, str):
        if throw:
            raise ValueError(f'Invalid country: {country}')
        return None

    try:
        return pycountry.countries.lookup(country)
    except LookupError:
        if country in country_iso_special_cases:
            return country_iso_special_cases[country]

        if fuzzy:
            try:
                return pycountry.countries.search_fuzzy(country)[0]
            except LookupError:
                pass

        if throw:
            raise ValueError(f'Country not found: {country}')

        return None

class DFWrapper:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df
        lower_column_names(self.df)

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value

class GDELT(DFWrapper):

    def __init__(self, df):
        super().__init__(df)
        self.add_country_names()

    def add_country_names(self):
        unique_codes = self.df['actor1country'].unique().tolist()
        unique_codes.extend(self.df['actor2country'].unique().tolist())
        unique_codes = set(unique_codes)

        code_to_name = {}
        for code in unique_codes:
            try :
                country = pycountry.countries.lookup(code)
                code_to_name[code] = country.name
            except LookupError:
                pass

        self.df = self.df[
            self.df['actor1country'].isin(code_to_name) &
            self.df['actor2country'].isin(code_to_name)
        ].copy()

        self.df['actor1country_name'] = self.df['actor1country'].apply(lambda x: code_to_name[x])
        self.df['actor2country_name'] = self.df['actor2country'].apply(lambda x: code_to_name[x])

    def clean_data(self, percentile: float = 0.95):
        df = self.df[self.df['actor1country'] != self.df['actor2country']]
        df.dropna(inplace=True)
        percentile = self.df.groupby('year')['sum_nummentions'].transform(lambda x: x.quantile(percentile))
        self.df = self.df[self.df['sum_nummentions'] >= percentile]
        return self

class Migration(DFWrapper):

    iso_or_codes: Optional[List[str]] = None
    iso_des_codes: Optional[List[str]] = None

    def __init__(self, df):
        super().__init__(df)
        self.translate_codes()

    def list_iso_or_codes(self):
        if self.iso_or_codes:
            return self.iso_or_codes
        codes = self.df['iso_or'].unique()
        valid = [code for code in codes if get_country_iso3(code, throw=False)]
        self.iso_or_codes = valid
        return valid

    def list_iso_des_codes(self):
        if self.iso_des_codes:
            return self.iso_des_codes
        codes = self.df['iso_des'].unique()
        valid = [code for code in codes if get_country_iso3(code, throw=False)]
        self.iso_des_codes = valid

    def translate_codes(self):
        origins = self.df['origin'].unique()
        destinations = self.df['destination'].unique()

        origin_lookup = {country: get_country_iso3(country, fuzzy=True, throw=False) for country in origins}
        destination_lookup = {country: get_country_iso3(country, fuzzy=True, throw=False) for country in destinations}

        self.df['iso_or'] = self.df['origin'].apply(lambda x: origin_lookup[x])
        self.df['iso_des'] = self.df['destination'].apply(lambda x: destination_lookup[x])

class Refugee(DFWrapper):

    iso_or_codes: Optional[List[str]] = None
    iso_des_codes: Optional[List[str]] = None

    def __init__(self, df):
        super().__init__(df)
        self.translate_to_iso()
        self.save_total()
        self.drop_unused()

    def translate_to_iso(self):
        origins = self.df['country of origin'].unique()
        asylums = self.df['country of asylum'].unique()

        origin_lookup = {country: get_country_iso3(country, fuzzy=True, throw=False) for country in origins}
        asylum_lookup = {country: get_country_iso3(country, fuzzy=True, throw=False) for country in asylums}

        self.df['iso_or'] = self.df['country of origin'].apply(lambda x: origin_lookup[x])
        self.df['iso_des'] = self.df['country of asylum'].apply(lambda x: asylum_lookup[x])

    def list_iso_or_codes(self):
        if self.iso_or_codes:
            return self.iso_or_codes
        codes = self.df['iso_or'].unique()
        valid = [code for code in codes if get_country_iso3(code, throw=False)]
        self.iso_or_codes = valid
        return valid

    def list_iso_des_codes(self):
        if self.iso_des_codes:
            return self.iso_des_codes
        codes = self.df['iso_des'].unique()
        valid = [code for code in codes if get_country_iso3(code, throw=False)]
        self.iso_des_codes = valid
        return valid

    def save_total(self):
        self.df['refugees'] = self.df['refugees under unhcr\'s mandate'] + self.df['asylum-seekers']

    def drop_unused(self):
        columns_to_drop = [
            'refugees under unhcr\'s mandate',
            'asylum-seekers',
            'returned refugees',
            'idps of concern to unhcr',
            'returned idpss',
            'stateless persons',
            'others of concern',
            'other people in need of international protection',
            'host community'
        ]
        self.df.drop(columns=columns_to_drop, inplace=True)
        
def load_country_centroids() -> Dict[str, Tuple[float, float]]:
    """
    Loads country centroids (average latitude and longitude) from a remote CSV file.
    Returns a dictionary mapping ISO3 country codes to (latitude, longitude) tuples.
    """
    url = 'https://cdn.jsdelivr.net/gh/gavinr/world-countries-centroids@v1/dist/countries.csv'
    df = pd.read_csv(url)
    country_centroids = {}
    for _, row in df.iterrows():
        iso3 = get_country_iso3(row['COUNTRY'], fuzzy=True, throw=False)
        if iso3 is None:
            continue
        lat = row['latitude']
        lon = row['longitude']
        country_centroids[iso3] = (lat, lon)
    return country_centroids

def get_gdelt(path: str) -> GDELT:
    return GDELT(load_df(path))

def get_migration(path: str) -> Migration:
    return Migration(load_df(path))

def get_refugee(path: str) -> Refugee:
    return Refugee(load_df(path))
