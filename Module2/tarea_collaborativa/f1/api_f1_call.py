import streamlit as st
import requests
import pandas as pd
import time


# Set your API key
API_BASE = "https://v1.formula-1.api-sports.io"
API_KEY = st.secrets["api_f1"]["API_F1_KEY"]

headers = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": "v1.formula-1.api-sports.io"
}

@st.cache_data
def api_get(endpoint, params=None, debug=False):
    resp = requests.get(f"{API_BASE}/{endpoint}", headers=headers, params=params)
    resp.raise_for_status()
    if debug:
        print(resp.text)
    return resp.json().get("response", [])

@st.cache_data
def get_races(endpoint='races', params=None):
    return api_get(endpoint, params=params)


@st.cache_data
def get_rankings_drivers(endpoint='rankings/drivers', params=None):
    return api_get(endpoint, params=params)


@st.cache_data
def get_rankings_teams(endpoint='rankings/teams', params=None):
    return api_get(endpoint, params=params)


@st.cache_data(show_spinner="Loading Super Time...")
def get_super_times_by_season(season: int) -> dict:

    def time_to_ms(t):
        minutes, seconds = t.split(":")
        return (int(minutes) * 60 + float(seconds)) * 1000

    """
    Calcola i Super Time per tutte le gare di una stagione.
    Rispetta il rate limit di 10 richieste al minuto.
    
    Restituisce:
        - races (DataFrame con le gare)
        - super_times (dict con i tempi e info per ogni gara)
    """
    super_times = {}
    
    races = pd.json_normalize(api_get("races", {"season": season, "type": "Race"}, debug=True))
    
    for i, (_, race) in enumerate(races.iterrows()):
        race_id = race["id"]
        race_name = race["competition.name"] + " - " + race["circuit.name"]
        
        try:
            best_laps = api_get("rankings/fastestlaps", {"race": race_id})
        except Exception as e:
            print(f"Errore nella richiesta per race {race_id}: {e}")
            continue
        
        # Delay tra richieste per rispettare il limite
        time.sleep(6.5)  # ≈ 9 richieste al minuto = sicuro
        
        best_laps_df = pd.json_normalize(best_laps)
        
        if not best_laps_df.empty:
            best_laps_df["time_ms"] = best_laps_df["time"].apply(time_to_ms)
            best_laps_df["superTimeRatio"] = best_laps_df["time_ms"] / best_laps_df["time_ms"].min()
            best_laps_df["superTimeDelta"] = best_laps_df["superTimeRatio"] - 1
            best_laps_df["superTimeDelta%"] = (best_laps_df["superTimeDelta"] * 100).round(2)

            super_times[race_id] = {
                "race_name": race_name,
                "data": best_laps_df
            }

    return races, super_times