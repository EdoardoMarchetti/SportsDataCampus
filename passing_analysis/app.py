from viz import create_pitch_figure, passing_map, passing_map_mpl, plot_pass_network
import streamlit as st
import pandas as pd
from mplsoccer import Sbopen


def prepare_data_for_passing_network(events, players, team):
    # Extract jersey numbers per player
    jersey_data = players[['player_id', 'jersey_number']].drop_duplicates()

    # Merge event data with jersey numbers using player_id
    df = pd.merge(events, jersey_data, on='player_id')

    # Add column with passer jersey number
    df['passer'] = df['jersey_number']

    # Filter only events from the specified team
    df = df[df['team_name'] == team]

    # Keep only pass events
    passes = df[df['type_name'] == 'Pass']

    # Filter only completed passes (missing outcome_name)
    successful = passes[passes['outcome_name'].isnull()]

    # Convert pass recipient IDs to integers
    rec = pd.to_numeric(successful['pass_recipient_id'], downcast='integer')

    # Rename columns to match recipient ID and jersey number
    jersey_data = jersey_data.rename(columns={
        'player_id': 'pass_recipient_id',
        'jersey_number': 'pass_recipient'
    })

    # Merge successful passes with recipient jersey numbers
    successful = pd.merge(df, jersey_data, on='pass_recipient_id')

    # Extract minute of the first substitution
    subs = df[df['type_name'] == 'Substitution']['minute']
    firstSub = subs.min()

    # Keep only passes made before the first substitution
    successful = successful[successful['minute'] < firstSub]

    # Compute average location and count of passes for each passer
    average_locations = successful.groupby('passer').agg({'x': ['mean'], 'y': ['mean', 'count']})
    average_locations.columns = ['x', 'y', 'count']

    # Count number of passes between each pair of players
    pass_between = successful.groupby(['passer', 'pass_recipient']).id.count().reset_index()
    pass_between = pass_between.rename(columns={'id': 'pass_count'})

    # Add average locations for both passer and recipient
    pass_between = pass_between.merge(average_locations, left_on='passer', right_index=True)
    pass_between = pass_between.merge(average_locations, left_on='pass_recipient', right_index=True,
                                       suffixes=['', '_end'])

    # Drop duplicates just in case
    pass_between.drop_duplicates(inplace=True)

    # Return the passing network data
    return pass_between, average_locations


st.set_page_config(layout="wide")
st.title("📊 Passing Analysis")

# Load the data
parser = Sbopen()
competitions_df = parser.competition()

# Create horizontal selectors using columns
st.subheader("Select competition")
col1, col2, col3, col4 = st.columns(4)

with col1:
    country = st.selectbox("Country", competitions_df['country_name'].unique())

filtered_by_country = competitions_df[competitions_df['country_name'] == country]

with col2:
    competition = st.selectbox(
        "Competition", 
        filtered_by_country['competition_name'].unique()
        if not filtered_by_country.empty else [])

filtered_by_competition = filtered_by_country[
    filtered_by_country['competition_name'] == competition
]

with col3:
    season = st.selectbox(
        "Season", 
        filtered_by_competition['season_name'].unique()
        if not filtered_by_competition.empty else [])

filtered_by_season = filtered_by_competition[
    filtered_by_competition['season_name'] == season
]

with col4:
    gender = st.selectbox(
        "Gender", 
        filtered_by_season['competition_gender'].unique()
        if not filtered_by_season.empty else [])

# Final filtering
filtered = filtered_by_season[filtered_by_season['competition_gender'] == gender]

# Extract competition_id and season_id
if filtered.empty:
    st.warning("No competitions match the selected filters.")
    st.stop()

competition_id = filtered.iloc[0]['competition_id']
season_id = filtered.iloc[0]['season_id']

# Load matches
matches_df = parser.match(competition_id, season_id)

# Match label formatting
def format_match(row):
    return (
        f"{row['kick_off']} | gw {row['match_week']} | "
        f"{row['home_team_name']} - {row['away_team_name']} | "
        f"{row['home_score']} - {row['away_score']}"
    )

matches_df['label'] = matches_df.apply(format_match, axis=1)
matches_df.sort_values(by='match_week', inplace=True)


# Match selector
st.subheader("Select match")
selected_match_label = st.selectbox("Match", matches_df['label'])
home_team = selected_match_label.split('|')[2].split('-')[0].strip()
away_team = selected_match_label.split('|')[2].split('-')[1].strip()
teams = [home_team, away_team]

# Extract match_id
selected_match = matches_df[matches_df['label'] == selected_match_label].iloc[0]
match_id = selected_match['match_id']

# Load events
with st.spinner("Loading match events..."):
    events, related, freeze, players = parser.event(match_id)
    lineup = parser.lineup(match_id)



#MARK: PAssing map section


cols = st.columns(2)

for team, col in zip(teams, cols):

    # This section was copy pasted from other forums to reduce white space and hide header and footer
    st.markdown("""
            <style>
                .stPlotlyChart {
                    height: 90vh !important;
                }
            </style>
            """, unsafe_allow_html=True)

    with col:
        df_passes = events[(events['type_name'] == 'Pass') & (events['team_name'] == team)].copy()

        if df_passes.empty:
            st.warning("No passes found in this match.")
            st.stop()

        #fig = create_pitch_figure()
        fig = passing_map(df_passes, title=team, sub_title=selected_match_label)
        st.plotly_chart(fig, use_container_width = False)


        pass_between, average_locations = prepare_data_for_passing_network(events, players, team=team)
        # Suppose pass_between and avg_loc are your DataFrames
        fig = plot_pass_network(pass_between, average_locations, pitch_bg='white', node_fill='lightgray', team_name=team, lineup=lineup,
                                title=team+' - Passing Network', sub_title=selected_match_label)
        
        st.plotly_chart(fig, use_container_width = False)


        # Selezione del giocatore
        selectionable_players = df_passes['player_name'].unique()
        selected_player = st.selectbox("Select Player", selectionable_players)

        # Filtra i passaggi per il giocatore selezionato
        df_player_passes = df_passes[df_passes['player_name'] == selected_player]


        fig = passing_map_mpl(df_player_passes, title=selected_player, sub_title=f"{selected_match_label}")
        st.pyplot(fig)


        








