import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st



def donut_side_chart(df_fixtures, side='home'):
    # --- Prepara i dati come nel codice Plotly ---
    # Calcolo W/D/L per team in casa
    home = df_fixtures.groupby(f'teams.{side}.name')[f'teams.{side}.winner'] \
            .value_counts().unstack().fillna(0)
    home.columns = ['Loss','Win']  # attenzione: vero/falso → Win/Loss
    home['Draw'] = df_fixtures[df_fixtures[f'teams.{side}.winner'].isnull()] \
                .groupby(f'teams.{side}.name').size()
    home = home.fillna(0)  # in caso di team senza pareggi
    home = home[['Win','Draw','Loss']].reset_index()

    # --- Parametri di layout ---
    teams = sorted(home[f'teams.{side}.name'].unique())
    teams_per_row = 4
    n_rows = int(np.ceil(len(teams) / teams_per_row))

    fig, axes = plt.subplots(n_rows, teams_per_row, figsize=(teams_per_row * 2.5, n_rows * 2.5))
    axes = axes.flatten()

    # --- Loop per ogni team ---
    for idx, row in home.iterrows():
        ax = axes[idx]
        team = row[f'teams.{side}.name']
        values = [row['Win'], row['Draw'], row['Loss']]
        total = sum(values)
        labels = ['Win', 'Draw', 'Loss']
        colors = {'Win': '#2ecc71', 'Draw': '#f1c40f', 'Loss': '#e74c3c'}

        wedges, texts = ax.pie(
            values,
            startangle=90,
            colors=[colors[l] for l in labels],
            wedgeprops={'width': 0.25, 'edgecolor': 'white'},
            radius=0.7
        )

        # Aggiungi etichette con % e valore assoluto
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.55 * np.cos(np.deg2rad(angle))
            y = 0.55 * np.sin(np.deg2rad(angle))
            pct = values[i] / total * 100
            ax.text(x, y, f"{pct:.0f}%\n({int(values[i])})", ha='center', va='center', fontsize=8, color='black')

        # Calcola punti (3 x vittorie + 1 x pareggi)
        points = int(row['Win'] * 3 + row['Draw'] * 1)
        ax.text(0, 0, f"{points}\nPoints", ha='center', va='center', fontsize=14, weight='bold', color='black')

        ax.set_title(team, fontsize=14)
        ax.axis('equal')

    # Nasconde eventuali subplot vuoti
    for j in range(len(teams), len(axes)):
        fig.delaxes(axes[j])

    legend_patches = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    fig.legend(handles=legend_patches,
            loc='upper center',
            ncol=3,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 1))

    plt.suptitle(f"{side.capitalize()} Results and Points per Team", fontsize=20, y=1.02)
    plt.tight_layout()
    
    return fig, ax


# Prepara df gol fatti/subiti
def goal_scored_vs_conceeded(df_fixtures):
    home = df_fixtures[['teams.home.name','goals.home','goals.away', 'teams.away.name']].copy()
    home.columns = ['team','scored','conceded', 'opponent']
    away = df_fixtures[['teams.away.name','goals.away','goals.home', 'teams.home.name']].copy()
    away.columns = ['team','scored','conceded', 'opponent']
    goals = pd.concat([home, away])

    fig = px.bar(goals, x='team', y=['scored','conceded'],
                barmode='group',
                hover_data=['opponent'],
                title="Goals Scored vs Conceded per Team")

    return fig


def cumulative_points(df_fixtures):
    # Calcola punti per match
    def points(row, side):
        if row[f'teams.{side}.winner'] == True: return 3
        if row[f'teams.{side}.winner'] == False: return 0
        return 1

    # Unisci e ordina per data
    home = df_fixtures[['fixture.date','teams.home.name']].copy()
    home['points'] = df_fixtures.apply(lambda r: points(r,'home'), axis=1)
    home.columns = ['date','team','points']
    away = df_fixtures[['fixture.date','teams.away.name']].copy()
    away['points'] = df_fixtures.apply(lambda r: points(r,'away'), axis=1)
    away.columns = ['date','team','points']

    points_df = pd.concat([home,away]).sort_values('date')
    points_df['cum_points'] = points_df.groupby('team')['points'].cumsum()

    fig = px.line(points_df, x='date', y='cum_points', color='team',
                title="Cumulative Points per Team over Time")
    return fig


def home_v_away_wins(df_fixtures, n_teams):
    # Calcola pct vittorie casa e trasferta
    home_perf = (df_fixtures.groupby('teams.home.name')['teams.home.winner'] \
                .sum().rename('home_win_pct') / (n_teams-1)) * 100
    away_perf = ((df_fixtures.groupby('teams.away.name')['teams.away.winner'] \
                .sum().rename('away_win_pct') / (n_teams-1))) * 100
    perf = pd.concat([home_perf, away_perf], axis=1).reset_index()

    fig = px.scatter(perf, x='home_win_pct', y='away_win_pct', text='index',
                    title="Home vs Away Win %", labels={
                        'home_win_pct':'Home Win %',
                        'away_win_pct':'Away Win %'
                    })
    fig.update_traces(textposition='top center')
    return fig


def win_per_weekday_distribution(df_fixtures):
    df_fixtures['weekday'] = pd.to_datetime(df_fixtures['fixture.date']).dt.day_name()
    home_w = df_fixtures[df_fixtures['teams.home.winner'] == True].groupby('weekday').size()
    away_w = df_fixtures[df_fixtures['teams.away.winner'] == True].groupby('weekday').size()
    win_df = pd.concat([home_w, away_w], axis=1).fillna(0)
    win_df.columns = ['home_wins', 'away_wins']

    win_df = win_df.reset_index().melt(id_vars='weekday', var_name='result', value_name='wins')



    fig = px.bar(
        win_df, x='weekday', y='wins', color='result', barmode='group',
        category_orders={'weekday': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']},
        title="Wins per Weekday (Home vs Away)",
        text='wins'  # ← Etichette sulle barre
    )

    fig.update_traces(textposition='outside')  # Posiziona le etichette sopra le barre

    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis_title="Number of Wins",
        xaxis_title="Weekday"
    )

    return fig




def team_trend_analysis(df_fixtures, team):
    # Mappa il risultato in punti (3 per vittoria, 1 per pareggio, -3 per sconfitta)
    def calculate_points(row, side):
        if side == 'home':
            if row[f'teams.{side}.winner'] == True: 
                return 3
            elif row[f'teams.{side}.winner'] == False: 
                return 0
            else:
                return 1  # pareggio
        else:
            if row[f'teams.{side}.winner'] == True: 
                return 3
            elif row[f'teams.{side}.winner'] == False: 
                return 0
            else:
                return 1  # pareggio

    # Creiamo un nuovo dataframe con i risultati in punti per casa (home) e trasferta (away)
    home_games = df_fixtures.loc[(df_fixtures[f'teams.home.name'] == team), ['fixture.date', 'teams.home.name', 'teams.away.name', 'teams.home.winner', 'goals.home', 'goals.away']].copy()
    home_games['home_away'] = 'H'
    home_games['home_points'] = home_games.apply(lambda row: calculate_points(row, 'home'), axis=1)
    home_games['label'] = home_games['teams.home.name'] + ' vs ' + home_games['teams.away.name'] + ' | ' + home_games['goals.home'].astype(str) + '-' + home_games['goals.away'].astype(str)
    home_games = home_games.rename(columns={'teams.home.name': 'team', 'teams.away.name': 'opponent', 'home_points': 'points'})

    away_games = df_fixtures.loc[(df_fixtures[f'teams.away.name'] == team), ['fixture.date', 'teams.away.name', 'teams.home.name',  'teams.away.winner', 'goals.home', 'goals.away']].copy()
    away_games['home_away'] = 'A'
    away_games['away_points'] = away_games.apply(lambda row: calculate_points(row, 'away'), axis=1)
    away_games['label'] = away_games['teams.home.name'] + ' vs ' + away_games['teams.away.name'] + ' | ' + away_games['goals.home'].astype(str) + '-' + away_games['goals.away'].astype(str)
    away_games = away_games.rename(columns={'teams.away.name': 'team', 'teams.home.name': 'opponent', 'away_points': 'points'})

    # Uniamo i dati delle partite in casa e in trasferta
    all_games = pd.concat([home_games, away_games])

    # Aggiungiamo i colori in base al risultato
    result_to_color = {3: 'green', 1: 'yellow', 0: 'red'}
    result_to_level = {3: 3, 1: 1, 0: 0.1}
    level_to_color = {3: 'green', 1:  'yellow', 0.1: 'red'}
    
    all_games['color'] = all_games['points'].map(result_to_color)
    all_games['level'] = all_games['points'].map(result_to_level)
    all_games = all_games.sort_values(by='fixture.date')
    all_games['level'] = pd.Categorical(all_games['level'])
    
    # Creiamo il grafico a barre
    fig = px.bar(all_games, 
                x='fixture.date', 
                y='level', 
                color='level', 
                title=f'Match Results (Points) for {team}',
                labels={'home_away': 'Home/Away', 'points': 'Points'},
                color_discrete_map=level_to_color,
                hover_data={'opponent': True, 'points': True, 'fixture.date': True, 'team': False, 'label': True, 'level':False},
                )

    # Personalizziamo il layout
    fig.update_layout(
        xaxis_title="Match Date",
        yaxis_title="",
        hovermode='x unified',  # Unified hover for better visibility
        showlegend=False
    )

    fig.update_yaxes(title='', visible=False, showticklabels=False)

    # Mostriamo il grafico
    return fig



