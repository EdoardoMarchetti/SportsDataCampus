# pages/1_Season_Analysis.py

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import requests

# Funzione per richiamare l’API (già disponibile)
from data_viz import create_top10_table_image_f1, plot_super_time
from api_f1_call import * 
from f1_data_preprocessing import build_super_time_dataframe 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Season Analysis", layout="wide")

# --- TITLE ---
st.title("🏎️ Season Analysis")

# --- SEASON SELECTOR ---
season = st.selectbox("Select a season", [2021, 2022, 2023])


# --- KPI: Number of Races ---
races_data = get_races("races", {"season": season, "type": "Race"})
num_races = len(races_data)

# --- KPI: Winning Driver ---
drivers_rankings = get_rankings_drivers("rankings/drivers", {"season": season})
winner_driver = drivers_rankings[0]

# --- KPI: Winning Team ---
teams_rankings = get_rankings_teams("rankings/teams", {"season": season})
winner_team = teams_rankings[0]

# --- KPI LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    gp_html = f"""
    <style>
        .gp-card {{
            background-color: #f9f9f9;
            border: 1px solid #000000;
            border-radius: 15px;
            padding: 15px 20px;
            margin-top: 10px;
            max-width: 450px;
            min-height: 130px; 
            box-sizing: border-box;
            font-family: Arial, sans-serif;
            margin-left: auto;
            margin-right: auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .gp-card .title {{
            font-weight: bold;
            font-size: 18px;
            color: #333;
            margin-bottom: 12px;
        }}
        .gp-info-row {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .gp-info-row img {{
            width: 60px;
            height: 60px;
            border-radius: 10px;
            flex-shrink: 0;
        }}
        .gp-info {{
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .gp-info .name {{
            font-weight: bold;
            font-size: 16px;
            color: #333;
        }}
        .gp-info .location {{
            font-size: 14px;
            color: #666;
        }}
    </style>

    <div class="gp-card">
        <div class="gp-info-row">
            <div style="font-size: 32px;">🏁</div>
            <div class="gp-info">
                <div class="name">Grand Prix number</div>
                <div class="location">{num_races}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(gp_html, unsafe_allow_html=True)

with col2:
    with st.container():
        driver_name = winner_driver['driver']['name']
        driver_points = winner_driver['points']
        driver_img_url = winner_driver['driver']['image']
        
        driver_html = f"""
        <style>
            .driver-card {{
                background-color: #f9f9f9;
                border: 1px solid #000000;
                border-radius: 15px;
                padding: 15px 20px;
                margin-top: 10px;
                max-width: 450px;
                box-sizing: border-box;
                font-family: Arial, sans-serif;
                margin-left: auto;
                margin-right: auto;
                min-height: 130px; 
            }}
            .driver-card .title {{
                font-weight: bold;
                font-size: 18px;
                color: #333;
                margin-bottom: 12px;
            }}
            .driver-info-row {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .driver-info-row img {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                flex-shrink: 0;
            }}
            .driver-info {{
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            .driver-info .name {{
                font-weight: bold;
                font-size: 16px;
                color: #333;
            }}
            .driver-info .points {{
                font-size: 14px;
                color: #666;
            }}
        </style>

        <div class="driver-card">
            <div class="title">🥇 Driver Champion</div>
            <div class="driver-info-row">
                <img src="{driver_img_url}" alt="Driver Image" />
                <div class="driver-info">
                    <div class="name">{driver_name}</div>
                    <div class="points">{driver_points} points</div>
                </div>
            </div>
        </div>
        """

        st.markdown(driver_html, unsafe_allow_html=True)

with col3:
    team_name = winner_team['team']['name']
    team_points = winner_team['points']
    team_img_url = winner_team['team']['logo']
    
    team_html = f"""
        <style>
            .team-card {{
                background-color: #f9f9f9;
                border: 1px solid #000000;
                border-radius: 15px;
                padding: 15px 20px;
                margin-top: 10px;
                max-width: 450px;
                box-sizing: border-box;
                font-family: Arial, sans-serif;
                margin-left: auto;
                margin-right: auto;
                min-height: 130px; 
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            .team-card .title {{
                font-weight: bold;
                font-size: 18px;
                color: #333;
                margin-bottom: 12px;
            }}
            .team-info-row {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .team-info-row img {{
                width: 120px;
                height: 60px;
                border-radius: 10px;
                flex-shrink: 0;
            }}
            .team-info {{
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            .team-info .name {{
                font-weight: bold;
                font-size: 16px;
                color: #333;
            }}
            .team-info .points {{
                font-size: 14px;
                color: #666;
            }}
        </style>

        <div class="team-card">
            <div class="title">🏆 Winner Team</div>
            <div class="team-info-row">
                <img src="{team_img_url}" alt="Team Image" />
                <div class="team-info">
                    <div class="name">{team_name}</div>
                    <div class="points">{team_points} points</div>
                </div>
            </div>
        </div>
        """
    st.markdown(team_html, unsafe_allow_html=True)


st.write("")

drivers, teams = st.columns(2)


with drivers:
    df_drivers = pd.DataFrame([
    {
        "rank": d["position"],
        "name": d["driver"]["name"],
        "imageDataURL": d["driver"]["image"],
        "team_name": d["team"]["name"],
        "team_logo": d["team"]["logo"],
        "points": float(d["points"]),
        "driver_id": d["driver"]["id"],
    }
    for d in drivers_rankings[:10]
    ])

    


    fig = create_top10_table_image_f1(
        df_drivers,
        selected_id=None,
        id_column="driver_id",
        metric_visible_name="Drivers",
        img_width=0.35
    )

    st.pyplot(fig)


with teams:
    df_teams = pd.DataFrame([
        {
            "rank": t["position"],
            "name": t["team"]["name"],
            "imageDataURL": t["team"]["logo"],
            "points": t["points"],
            "team_id": t["team"]["id"],
        }
        for t in teams_rankings
    ])

    fig = create_top10_table_image_f1(
        df_teams,
        selected_id=None,
        id_column="team_id",
        metric_visible_name="Team",
        img_width=0.2
    )

    st.pyplot(fig)


super_by = st.selectbox(
    label='Super times by:',
    options = ['Driver', 'Team'],
    index = 0
)


races, super_times = get_super_times_by_season(season)
super_time_df = build_super_time_dataframe(races, super_times=super_times, by=super_by.lower())

fig = plot_super_time(df=super_time_df, by=super_by.lower())
st.plotly_chart(fig)