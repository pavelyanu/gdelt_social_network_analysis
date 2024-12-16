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
from newspaper import Article as NewspaperArticle

from waybackpy import WaybackMachineAvailabilityAPI

import nltk
nltk.download('punkt_tab')

"""
GDELT HEAD:
head events/data/gdelt_top100_events_1960_2020_full.csv 
GLOBALEVENTID,Actor1CountryCode,Actor2CountryCode,Actor1Name,Actor2Name,Year,SQLDATE,EventCode,NumMentions,GoldsteinScale,AvgTone,ActionGeo_Lat,ActionGeo_Long,percentile,SOURCEURL
355098,ABW,NLD,ARUBA,NETHERLANDS,1979,19791123,051,4,3.4,5,12.1911,-68.2567,1,unspecified
1433649,ABW,NLD,ARUBA,THE NETHERLAND,1981,19811028,037,10,5,3.1413612565445,12.25,-68.75,1,unspecified
3684442,ABW,,ARUBA,,1984,19841004,043,10,2.8,4.9645390070922,12.5167,-70.0333,1,unspecified
5592722,ABW,NLD,ARUBA,THE NETHERLAND,1986,19861126,193,19,-10,4.76582068155965,12.5,-69.9667,1,unspecified
6940533,ABW,COL,ARUBA,COLOMBIAN,1988,19880514,043,9,2.8,2.54777070063694,12.5,-69.9667,1,unspecified

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

class Tone(Enum):
    POSITIVE = 0
    NEGATIVE = 1

class DFWrapper:
    df: pd.DataFrame
    min_year: int
    max_year: int

    def __init__(self, df: pd.DataFrame):
        self.df = df
        lower_column_names(self.df)
        self.min_year = self.df['year'].min()
        self.max_year = self.df['year'].max()

    def check_year(self, year: int) -> None:
        if year < self.min_year or year > self.max_year:
            raise ValueError(f'Year out of range: {year}. Please provide a year between {self.min_year} and {self}')

    def year(self, year: int) -> pd.DataFrame:
        return self.df[self.df['year'] == year]

    def year_mask(self, year: int) -> bool:
        return self.df['year'] == year

    def range_mask(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> bool:
        if start_year and end_year:
            return (self.df['year'] >= start_year) & (self.df['year'] <= end_year)
        if start_year:
            return self.df['year'] >= start_year
        if end_year:
            return self.df['year'] <= end_year
        return True

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value

    def top_k(self, compare_by: str, k: int, masks: Optional[List] = None) -> pd.DataFrame:
        if masks is None or not masks:
            return self.df.nlargest(k, compare_by)
        mask = np.logical_and.reduce(masks)
        return self.df[mask].nlargest(k, compare_by)

    def apply_masks(self, masks: List) -> pd.DataFrame:
        mask = np.logical_and.reduce(masks)
        return self.df[mask]


class GDELT(DFWrapper):

    iso_codes: Optional[List[str]] = None

    def __init__(self, df):
        super().__init__(df)

    def list_iso_codes(self):
        if self.iso_codes:
            return self.iso_codes
        codes = self.df['actor1countrycode'].unique()
        valid = [code for code in codes if get_country_iso3(code, throw=False)]
        self.iso_codes = valid
        return valid

    def country(self, country: str) -> pd.DataFrame:
        iso3 = get_country_iso3(country)
        return self.df[(self.df['actor1countrycode'] == iso3) | (self.df['actor2countrycode'] == iso3)]

    def country_year(self, country: str, year: int) -> pd.DataFrame:
        self.check_year(year)
        iso3 = get_country_iso3(country)
        return self.df[((self.df['actor1countrycode'] == iso3) | (self.df['actor2countrycode'] == iso3)) & (self.df['year'] == year)]

    def median_tone_in_country(self, country) -> float:
        iso3 = get_country_iso3(country)
        return self.df[(self.df['actor1countrycode'] == iso3) | (self.df['actor2countrycode'] == iso3)]['avgtone'].median()

    def actor1country_mask(self, country: str) -> bool:
        iso3 = get_country_iso3(country)
        return self.df['actor1countrycode'] == iso3

    def actor2country_mask(self, country: str) -> bool:
        iso3 = get_country_iso3(country)
        return self.df['actor2countrycode'] == iso3

    def year_mask(self, year: int) -> bool:
        return self.df['year'] == year

    def positive_tone_mask(self, tone: float) -> bool:
        return self.df['avgtone'] > tone

    def negative_tone_mask(self, tone: float) -> bool:
        return self.df['avgtone'] < tone

    def goldstein_mask(self, scale: float) -> bool:
        return self.df['goldsteinscale'] > scale

    def valid_url_mask(self) -> bool:
        return self.df['sourceurl'].apply(url_valid)

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

    def origin_mask(self, country_of_origin: str) -> bool:
        iso3 = get_country_iso3(country_of_origin)
        return self.df['iso_or'] == iso3

    def destination_mask(self, country_of_destination: str) -> bool:
        iso3 = get_country_iso3(country_of_destination)
        return self.df['iso_des'] == iso3

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

    def year_mask(self, year: int) -> bool:
        return self.df['year'] == year

    def range_mask(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> bool:
        if start_year and end_year:
            return (self.df['year'] >= start_year) & (self.df['year'] <= end_year)
        if start_year:
            return self.df['year'] >= start_year
        if end_year:
            return self.df['year'] <= end_year
        return True

    def origin_mask(self, country_of_origin: str) -> bool:
        iso3 = get_country_iso3(country_of_origin)
        return self.df['country of origin'] == iso3

    def asylum_mask(self, country_of_asylum: str) -> bool:
        iso3 = get_country_iso3(country_of_asylum)
        return self.df['country of asylum'] == iso3

@dataclass
class Article:
    url: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    text: Optional[str] = None
    top_image: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def download_and_parse(self, include_nlp: bool = False):
        article = NewspaperArticle(self.url)
        try:
            article.download()
            article.parse()
            self.title = article.title
            self.authors = article.authors
            self.text = article.text
            self.top_image = article.top_image
            if include_nlp:
                article.nlp()
                self.summary = article.summary
                self.keywords = article.keywords
        except Exception as e:
            print(f"Failed to process article at {self.url}: {e}")

class DownloadStatus(Enum):
    SUCCESS = 0
    NO_URL = 1
    INVALID_URL = 2
    NOT_TRIED = 3


@dataclass
class GDELTEvent:
    # GDELT fields
    globaleventid: int
    actor1countrycode: Optional[str]
    actor2countrycode: Optional[str]
    actor1name: Optional[str]
    actor2name: Optional[str]
    year: int
    sqldate: int
    eventcode: int
    nummentions: int
    goldsteinscale: float
    avgtone: float
    actiongeo_lat: float
    actiongeo_long: float
    percentile: float
    sourceurl: Optional[str]

    # Additional fields
    eventcodeexplanation: Optional[str] = None
    actor1countryname: Optional[str] = None
    actor2countryname: Optional[str] = None
    color: Optional[str] = None

    # Source Article related feilds
    article_download_status: DownloadStatus = DownloadStatus.NOT_TRIED
    article_url: Optional[str] = None
    article_title: Optional[str] = None
    article_authors: List[str] = field(default_factory=list)
    article_top_image: Optional[str] = None
    article_summary: Optional[str] = None
    article_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = self.__dict__.copy()
        result['article_download_status'] = self.article_download_status.value
        # This is so bad... somebody help...
        keywords_invalid = pd.isna(self.article_keywords)
        keywords_invalid = keywords_invalid if isinstance(keywords_invalid, bool) else keywords_invalid.all()
        if not keywords_invalid:
            result['article_keywords'] = '|'.join(self.article_keywords)
        return result

    @staticmethod
    def from_dict(data: Dict) -> 'GDELTEvent':
        if 'article_download_status' in data:
            data['article_download_status'] = DownloadStatus(data['article_download_status'])
        if 'article_keywords' in data and not pd.isna(data['article_keywords']):
            data['article_keywords'] = data['article_keywords'].split('|')
        return GDELTEvent(**data)

    def download_article(self, use_wayback: bool = False):
        if not url_valid(self.sourceurl):
            self.article_download_status = DownloadStatus.NO_URL
            return

        article = NewspaperArticle(self.sourceurl)
        try:
            article.download()
            article.parse()
            article.nlp()
        except Exception as e:
            if use_wayback:
                availability_api = WaybackMachineAvailabilityAPI(self.sourceurl)
                wayback_url = availability_api.oldest()
                article = NewspaperArticle(wayback_url)
                try:
                    article.download()
                except Exception as e:
                    self.article_download_status = DownloadStatus.INVALID_URL
                    return
            self.article_download_status = DownloadStatus.INVALID_URL
            return

        self.article_title = article.title
        self.article_authors = article.authors
        self.article_top_image = article.top_image
        self.article_summary = article.summary
        self.article_keywords = article.keywords

        self.article_download_status = DownloadStatus.SUCCESS

GDELTEventFields = list(GDELTEvent.__dataclass_fields__.keys())


class GDELTEventDatabase:
    path: str
    database: pd.DataFrame
    lock: threading.Lock

    def __init__(self, path: str, drop: bool = False):
        self.path = get_abs(path)
        if drop and os.path.exists(self.path):
            os.remove(self.path)
        self.initialize_database()
        self.lock = threading.Lock()

    def initialize_database(self):
        if not os.path.exists(self.path):
            self.database = pd.DataFrame(columns=GDELTEvent.__dataclass_fields__.keys())
        else:
            self.database = self.load_database(self.path)
        if 'globaleventid' in self.database.columns:
            self.database.set_index('globaleventid', inplace=True)
        else:
            self.database.index.name = 'globaleventid'

    @staticmethod
    def load_database(path: str) -> pd.DataFrame:
        return load_df(path)

    def save_database(self):
        with self.lock:
            print("Saving database...")
            self.database.to_csv(self.path, index=True)

    def extend(self, events: List[GDELTEvent]):
        new_data = pd.DataFrame([event.to_dict() for event in events])
        if new_data.empty:
            return
        new_data = new_data.drop_duplicates(subset='globaleventid')
        new_data.set_index('globaleventid', inplace=True)
        with self.lock:
            self.database.update(new_data)
            new_entries = new_data.index.difference(self.database.index)
            if not new_entries.empty:
                self.database = pd.concat([self.database, new_data.loc[new_entries]], sort=False)
            self.database.index.name = 'globaleventid'

    def contains(self, eventID: Union[int, str]) -> bool:
        eventID = int(eventID)
        with self.lock:
            return eventID in self.database.index

    def lookup(self, eventID: Union[int, str]) -> Optional[GDELTEvent]:
        eventID = int(eventID)
        with self.lock:
            if eventID in self.database.index:
                row = self.database.loc[eventID]
                if isinstance(row, pd.Series):
                    data = row.to_dict()
                    data['globaleventid'] = eventID
                    return GDELTEvent.from_dict(data)
                elif isinstance(row, pd.DataFrame):
                    data = row.iloc[0].to_dict()
                    data['globaleventid'] = eventID
                    return GDELTEvent.from_dict(data)
            return None

    def lookup_batch(self, eventIDs: pd.Series) -> pd.DataFrame:
        with self.lock:
            eventIDs = eventIDs.astype(int)
            existing_ids = self.database.index.intersection(eventIDs)
            result_df = self.database.loc[existing_ids]
            result_df = result_df.reset_index()
            return result_df

    def remove_duplicates(self):
        with self.lock:
            self.database = self.database[~self.database.index.duplicated(keep='last')]

def gdelt_list_to_df(gdelt_list: List[GDELTEvent]) -> pd.DataFrame:
    return pd.DataFrame([event.to_dict() for event in gdelt_list])

        
class AtomicCounter:
    lock: threading.Lock
    events_processed: int
    articles_loaded: int
    events_no_url: int
    events_invalid_url: int

    def __init__(self):
        self.lock = threading.Lock()
        self.events_processed = 0
        self.events_no_url = 0
        self.events_invalid_url = 0
        self.articles_loaded = 0

    def increment(self, processed: bool = False, loaded: bool = False, no_url: bool = False, invalid_url: bool = False):
        with self.lock:
            if processed:
                self.events_processed += 1
            if loaded:
                self.articles_loaded += 1
            if no_url:
                self.events_no_url += 1
            if invalid_url:
                self.events_invalid_url += 1

    def count(self, event: GDELTEvent):
        processed = True
        loaded = False
        no_url = event.article_download_status == DownloadStatus.NO_URL
        invalid_url = event.article_download_status == DownloadStatus.INVALID_URL
        self.increment(processed, loaded, no_url, invalid_url)

    def print_stats(self):
        with self.lock:
            print(f"Events processed: {self.events_processed}")
            print(f"Articles loaded: {self.articles_loaded}")
            print(f"Events with no URL: {self.events_no_url}")
            print(f"Events with invalid URL: {self.events_invalid_url}")

global_atomic_counter = AtomicCounter()

def process_gdelt_row(row: pd.Series, database: Optional[GDELTEventDatabase] = None) -> GDELTEvent:
    globaleventid = row.get('globaleventid')

    if database:
        if database.contains(globaleventid):
            event = database.lookup(globaleventid)
            if event:
                global_atomic_counter.increment(loaded=True)
                global_atomic_counter.count(event)
                return event

    data = row.to_dict()

    if 'globaleventid' not in data:
        data['globaleventid'] = globaleventid

    event = GDELTEvent.from_dict(data)

    event.color = 'green' if event.avgtone > 0 else 'red'
    event.eventcodeexplanation = CODE_MAPPING.get(event.eventcode, 'Unknown')
    event.download_article()

    null_name = "Unknown"

    actor1_country_name: Optional[Any] = get_country(row.get('actor1countrycode'), fuzzy=True, throw=False)
    event.actor1countryname = actor1_country_name.name if actor1_country_name else null_name

    actor2_country_name: Optional[Any] = get_country(row.get('actor2countrycode'), fuzzy=True, throw=False)
    event.actor2countryname = actor2_country_name.name if actor2_country_name else null_name

    global_atomic_counter.count(event)

    return event

def process_gdelt_df(df: pd.DataFrame, database: Optional[GDELTEventDatabase] = None, num_workers: int = 20) -> List[GDELTEvent]:
    """
    Process the GDELT DataFrame in parallel using ThreadPoolExecutor.
    """

    eventids = df['globaleventid']
    print("Total events: ", len(eventids))
    existing = []
    if database:
        loaded = database.lookup_batch(eventids)
        for _, l in loaded.iterrows():
            data = l.to_dict()
            e = GDELTEvent.from_dict(data)
            existing.append(e)

        loaded_ids = loaded['globaleventid']
        print("Loaded existing events: ", len(loaded_ids))
        df = df[~df['globaleventid'].isin(loaded_ids)]
        print("Remaining: ", len(df))

    events = []
    tasks = []

    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_gdelt_row, row, database): row for row in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing GDELT Events"):
            try:
                event = future.result()
                if event:
                    events.append(event)
            except Exception as e:
                print(f"Exception occurred: {e}")

    events.extend(existing)

    if database:
        database.extend(events)

    return events


@dataclass
class Arrow:
    destination: str
    origin: str
    origin_name: str
    destination_name: str
    weight: int
    color: Optional[str] = None

def process_migration_row(row) -> Tuple[Arrow, Arrow]:
    inflow = Arrow(
        destination=row['iso_des'],
        origin=row['iso_or'],
        weight=row['inflow'],
        origin_name='',
        destination_name='',
        color=COLOR_MAP['migration_in']
    )
    outflow = Arrow(
        destination=row['iso_des'],
        origin=row['iso_or'],
        weight=row['outflow'],
        origin_name='',
        destination_name='',
        color=COLOR_MAP['migration_out']
    )
    inflow_name_or: Optional[Any] = get_country(inflow.origin, throw=False)
    inflow.origin_name = inflow_name_or.name if inflow_name_or else "Unknown"

    inflow_name_des: Optional[Any] = get_country(inflow.destination, throw=False)
    inflow.destination_name = inflow_name_des.name if inflow_name_des else "Unknown"

    outflow_name_or: Optional[Any] = get_country(outflow.origin, throw=False)
    outflow.origin_name = outflow_name_or.name if outflow_name_or else "Unknown"

    outflow_name_des: Optional[Any] = get_country(outflow.destination, throw=False)
    outflow.destination_name = outflow_name_des.name if outflow_name_des else "Unknown"

    return inflow, outflow

def process_migration_df(df: pd.DataFrame, direction: str = "both") -> List[Arrow]:
    tuples = [process_migration_row(row) for _, row in df.iterrows()]
    if direction == "in":
        return [arrow[0] for arrow in tuples]
    if direction == "out":
        return [arrow[1] for arrow in tuples]
    return [arrow for pair in tuples for arrow in pair]

def process_refugee_row(row) -> Arrow:
    arrow = Arrow(
        destination=row['country of asylum'],
        origin=row['country of origin'],
        origin_name='',
        destination_name='',
        weight=row['refugees'],
        color=COLOR_MAP['refugee']
    )

    null_name = "Unknown"

    origin_name: Optional[Any] = get_country(arrow.origin, throw=False)
    arrow.origin_name = origin_name.name if origin_name else null_name

    destination_name: Optional[Any] = get_country(arrow.destination, throw=False)
    arrow.destination_name = destination_name.name if destination_name else null_name

    return arrow

def process_refugee_df(df: pd.DataFrame) -> List[Arrow]:
    return [process_refugee_row(row) for _, row in df.iterrows()]

@dataclass
class PlotUnit:
    """
    Contains information about an event and associated arrows
    """
    country: str
    event: GDELTEvent
    incoming_arrows: List[Arrow]
    outgoing_arrows: List[Arrow]

def generate_plot_data(
        gdelt: GDELT,
        migration: Migration,
        refugee: Refugee,
        span: Tuple[int, int],
        database: Optional[GDELTEventDatabase] = None,
        num_workers: int = 20) -> Dict[int, List[PlotUnit]]:

    data: Dict[int, List[PlotUnit]] = {}

    for year in range(span[0], span[1] + 1):
        yearly_data: List[PlotUnit] = []
        gdelt_year = gdelt.year(year)
        migration_year = migration.year(year)
        refugee_year = refugee.year(year)

        tasks = []
        for country in gdelt.list_iso_codes():
            iso_code = get_country_iso3(country, fuzzy=True)
            events = gdelt_year[gdelt_year['actor1countrycode'] == iso_code]
            migration_out = migration_year[migration_year['iso_or'] == iso_code]
            migration_in = migration_year[migration_year['iso_des'] == iso_code]
            refugee_out = refugee_year[refugee_year['country of origin'] == iso_code]
            refugee_in = refugee_year[refugee_year['country of asylum'] == iso_code]

            events = events[events['goldsteinscale'] >= 5]
            events = events.nlargest(10, 'nummentions')
            migration_out = migration_out.nlargest(10, 'outflow')
            migration_in = migration_in.nlargest(10, 'inflow')
            refugee_out = refugee_out.nlargest(10, 'refugees')
            refugee_in = refugee_in.nlargest(10, 'refugees')

            for _, event in events.iterrows():
                tasks.append((country, event, migration_in, refugee_in, migration_out, refugee_out))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_plot_unit, task, database): task for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing year {year}"):
                plot_unit = future.result()
                if plot_unit:
                    yearly_data.append(plot_unit)

        data[year] = yearly_data

    global_atomic_counter.print_stats()

    if database:
        database.extend([plot_unit.event for year_data in data.values() for plot_unit in year_data])

    return data

def process_plot_unit(task, database, retry: bool = False, success_only = True):
    country, event_row, migration_in, refugee_in, migration_out, refugee_out = task
    incoming_arrows = process_migration_df(migration_in, direction="in")
    incoming_arrows += process_refugee_df(refugee_in)
    outgoing_arrows = process_migration_df(migration_out, direction="out")
    outgoing_arrows += process_refugee_df(refugee_out)

    incoming_arrows = [arrow for arrow in incoming_arrows if arrow.destination]
    incoming_arrows = [arrow for arrow in incoming_arrows if arrow.origin]
    incoming_arrows = [arrow for arrow in incoming_arrows if arrow.weight > 0]

    outgoing_arrows = [arrow for arrow in outgoing_arrows if arrow.destination]
    outgoing_arrows = [arrow for arrow in outgoing_arrows if arrow.origin]
    outgoing_arrows = [arrow for arrow in outgoing_arrows if arrow.weight > 0]

    event = process_gdelt_row(event_row, database)
    if success_only:
        if event.article_download_status == DownloadStatus.SUCCESS:
            return PlotUnit(country, event, incoming_arrows, outgoing_arrows)
        else:
            return None
    return PlotUnit(country, event, incoming_arrows, outgoing_arrows)

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
