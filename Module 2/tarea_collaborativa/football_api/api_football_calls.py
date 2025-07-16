import streamlit as st
import requests


# Configurazione base
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": st.secrets["api_football"]["API_FOOTBALL_KEY"],
    "x-rapidapi-host": "v3.football.api-sports.io"
}




@st.cache_data
def get_countries():
    try:
        r = requests.get(f"{BASE_URL}/countries", headers=HEADERS)
        r.raise_for_status()  # Check if the request was successful
        return [c['name'] for c in r.json()['response']]
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching countries: {e}")
        st.stop()  # Stop the app

@st.cache_data
def get_leagues(country):
    try:
        r = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"country": country})
        r.raise_for_status()  # Check if the request was successful
        leagues = r.json()['response']
        return leagues
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching leagues for country {country}: {e}")
        st.stop()  # Stop the app

@st.cache_data
def get_seasons(league_id):
    try:
        r = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"id": league_id})
        r.raise_for_status()  # Check if the request was successful
        return [s['year'] for s in r.json()['response'][0]['seasons']]
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching seasons for league {league_id}: {e}")
        st.stop()  # Stop the app

@st.cache_data
def get_fixtures(league_id, season):
    try:
        r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"league": league_id, "season": season})
        r.raise_for_status()  # Check if the request was successful
        return r.json()['response']
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching fixtures for league {league_id} and season {season}: {e}")
        st.stop()  # Stop the app

@st.cache_data
def get_teams(league_id, year):
    try:
        r = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"league": league_id, "season": year})
        r.raise_for_status()  # Check if the request was successful
        return r.json()['response']
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching teams for league {league_id} and year {year}: {e}")
        st.stop()  # Stop the app
