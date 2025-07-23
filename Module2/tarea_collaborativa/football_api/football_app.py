from api_football_calls import *
from viz import *
import streamlit as st
from api_football_calls import get_countries, get_leagues, get_seasons, get_fixtures
import pandas as pd


st.set_page_config(layout="wide")
st.title("📊 Football Visual Analytics (API-Football)")

# --- SELEZIONE DINAMICA ---
country = st.selectbox("Select Country", get_countries())
league_df = pd.json_normalize(get_leagues(country))
league_name = st.selectbox("Select Competition", league_df['league.name'])
league_id = league_df.loc[league_df['league.name'] == league_name, 'league.id']

seasons = get_seasons(league_id)
season = st.selectbox("Select Season", seasons[::-1], index=1)

# --- OTTIENI FIXTURES ---
with st.spinner("Loading..."):
    fixtures_raw = get_fixtures(league_id, season)
    df_fixtures = pd.json_normalize(fixtures_raw)
    df_teams = pd.json_normalize(get_teams(league_id, season))

if df_teams.empty or df_fixtures.empty:
    st.error('No data available, please change selecion')
    st.stop()

st.header('Fixtures')
st.write(df_fixtures)

st.header('Teams')
st.write(df_teams)

# --- VISUALIZZAZIONI (basate su df_fixtures) ---
cols = st.columns(2, gap = "large")
with cols[0]:
    fig, ax = donut_side_chart(df_fixtures, side='home')
    st.pyplot(fig)
with cols[-1]:
    fig, ax = donut_side_chart(df_fixtures, side='away')
    st.pyplot(fig)



fig = cumulative_points(df_fixtures)
st.plotly_chart(fig)


fig = home_v_away_wins(df_fixtures, n_teams=len(df_teams))
st.plotly_chart(fig)


fig = goal_scored_vs_conceeded(df_fixtures)
st.plotly_chart(fig)

fig = win_per_weekday_distribution(df_fixtures)
st.plotly_chart(fig)

st.header("Team Trend")
team_selected = st.selectbox("Select team", options = df_teams['team.name'])
fig = team_trend_analysis(df_fixtures, team=team_selected)
st.plotly_chart(fig)

