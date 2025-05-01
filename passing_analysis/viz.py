import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
import math

def create_pitch_figure():
    """Crea un oggetto Figure con il campo da calcio disegnato (senza eventi)."""
    fig = go.Figure()

    # Configurazione layout campo
    fig.update_layout(
        # width=800, height=550,
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
        shapes=[
            # Bordo campo
            dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="black")),
            # Linea di metà campo
            dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="black")),
            # Cerchio centrale
            dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="black")),
            # Area grande sinistra
            dict(type="rect", x0=0, y0=20, x1=18, y1=60, line=dict(color="black")),
            # Area piccola sinistra
            dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="black")),
            # Area grande destra
            dict(type="rect", x0=102, y0=20, x1=120, y1=60, line=dict(color="black")),
            # Area piccola destra
            dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="black")),
        ],
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def passing_map(df, title: str, sub_title: str):
    """Create a passing map with team title, match label, and donut chart for pass outcomes."""
    
    def compute_angle(x0, y0, x1, y1):
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    # Separate passes
    df_completed = df[df['outcome_name'].isnull()]
    df_failed = df[df['outcome_name'].notnull()]
    total = len(df)

    # Subplot layout: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # title + donut
            [{"colspan": 2}, None],
        ],
        column_widths=[0.75, 0.15],
        row_heights=[0.15, 0.75],
        vertical_spacing=0.03
    )

    # === Title and subtitle ===
    fig.add_annotation(
        text=f"<b>{title}</b><br><span style='font-size:14px;color:gray'>{sub_title}</span>",
        xref="paper", yref="paper",
        x=0.1, y=0.975,
        showarrow=False,
        font=dict(size=20),
        align="left"
    )
    fig.update_layout(
        xaxis1=dict(visible=False),
        yaxis1=dict(visible=False)
    )

    # === Donut chart (top right) ===
    fig.add_trace(go.Pie(
        values=[len(df_completed), len(df_failed)],
        labels=["Completed", "Failed"],
        hole=0.75,
        marker=dict(colors=["green", "red"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
        showlegend=False,
        domain=dict(x=[0, 1.0], y=[0, 1.0])
    ), row=1, col=2)

    # Add total inside donut
    fig.add_annotation(
        text=f"<b>{total}</b>",
        font=dict(size=16),
        showarrow=False,
        xref="paper", yref="paper",
        x=0.9425, y=0.945
    )

    # === Pitch setup (bottom row) ===
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=2, col=1, autorange="reversed")

    pitch = create_pitch_figure()
    for shape in pitch.layout.shapes:
        fig.add_shape(shape, row=2, col=1)

    # Add passes
    for i, (_, r) in enumerate(df_completed.iterrows()):
        angle = compute_angle(r.x, r.y, r.end_x, r.end_y)
        hover = (
            f"From: ({r.x:.1f}, {r.y:.1f})<br>"
            f"To: ({r.end_x:.1f}, {r.end_y:.1f})<br>"
            f"Player: {r.player_name}<br>"
            f"Recipient: {r.pass_recipient_name}<br>"
            f"Outcome: completed"
        )
        fig.add_trace(go.Scatter(
            x=[r.x, r.end_x], y=[r.y, r.end_y],
            mode="lines",
            line=dict(color="green", width=2),
            hoverinfo="text", hovertext=hover,
            showlegend=(i == 0),  # only first gets legend
            name="Completed pass"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[r.end_x], y=[r.end_y],
            mode="markers",
            marker=dict( size=10, angle=angle, color="green"),
            hoverinfo="text", hovertext=hover,
            showlegend=False
        ), row=2, col=1)


    for i, (_, r) in enumerate(df_failed.iterrows()):
        hover = (
            f"From: ({r.x:.1f}, {r.y:.1f})<br>"
            f"To: ({r.end_x:.1f}, {r.end_y:.1f})<br>"
            f"Player: {r.player_name}<br>"
            f"Recipient: {r.pass_recipient_name}<br>"
            f"Outcome: {r.outcome_name}"
        )
        fig.add_trace(go.Scatter(
            x=[r.x, r.end_x], y=[r.y, r.end_y],
            mode="lines",
            line=dict(color="red", width=2),
            hoverinfo="text", hovertext=hover,
            showlegend=(i == 0),  # only first gets legend
            name="Failed"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[r.end_x], y=[r.end_y],
            mode="markers",
            marker=dict(symbol="x", size=10, color="red"),
            hoverinfo="text", hovertext=hover,
            showlegend=False
        ), row=2, col=1)


    # Add annotation note below the pitch
    fig.add_annotation(
        text="Note: Each pass ends at the marker symbol. Team attack from left to right",
        xref="paper", yref="paper",
        x=0.5, y=0, xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=12, color="gray")
    )


    fig.update_layout(
        autosize=False,
        width = 800,
        height=700,
        margin=dict(t=60, b=40, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.8,
            xanchor="center",
            x=0.5
        )
    )

    return fig




def passing_map_mpl(df, title: str, sub_title: str):
    """
    Draws a passing map using mplsoccer.Pitch, con titolo, sottotitolo,
    donut chart in alto a destra e nota in basso.
    """
    # ----------------------------
    # 1. Impostazioni generali
    # ----------------------------
    # Conta passaggi
    completed = df['outcome_name'].isnull().sum()
    failed    = len(df) - completed
    total     = len(df)

    # Crea figura e riquadri
    fig = plt.figure(figsize=(10, 8))
    # Pitch occupa la gran parte della figura
    ax_pitch = fig.add_axes([0, 0.05, 1, 0.9])
    # Donut chart in alto a destra
    ax_donut = fig.add_axes([0.82, 0.86, 0.15, 0.25])
    # Nota in basso
    ax_note = fig.add_axes([0, 0.01, 1, 0.05])
    ax_note.axis('off')

    # ----------------------------
    # 2. Disegna il campo
    # ----------------------------
    pitch = Pitch(pitch_type='statsbomb',# orientation='horizontal',
                  pitch_color='white', line_color='black')
    pitch.draw(ax=ax_pitch)

    # ----------------------------
    # 3. Disegna i passaggi
    # ----------------------------
    def compute_angle(x0, y0, x1, y1):
        return math.atan2(y1 - y0, x1 - x0)

    # Passaggi completati
    for _, r in df[df['outcome_name'].isnull()].iterrows():
        angle = compute_angle(r.x, r.y, r.end_x, r.end_y)
        pitch.arrows(r.x, r.y, r.end_x, r.end_y,
                     ax=ax_pitch, color='green', width=2,
                     headwidth=3, headlength=5, minlength=0.5, alpha=0.8)
        #ax_pitch.scatter(r.end_x, r.end_y, marker='^', color='green', s=50)

    # Passaggi falliti
    for _, r in df[df['outcome_name'].notnull()].iterrows():
        pitch.lines(r.x, r.y, r.end_x, r.end_y,
                    ax=ax_pitch, color='red', lw=2, alpha=0.8)
        ax_pitch.scatter(r.end_x, r.end_y, marker='x', color='red', s=50)

    # ----------------------------
    # 4. Titolo e sottotitolo
    # ----------------------------
    fig.suptitle(title, x=0.1, y=1.015, ha='left', fontsize=14)
    fig.text(0.1, 0.95, sub_title, ha='left', fontsize=10, color='gray')

    # ----------------------------
    # 5. Donut chart
    # ----------------------------
    sizes = [completed, failed]
    wedges, texts = ax_donut.pie(sizes, colors=['green', 'red'],
                                 startangle=90, wedgeprops=dict(width=0.25))
    ax_donut.set_aspect('equal')
    # Testo centrale
    ax_donut.text(0, 0, str(total), ha='center', va='center', fontsize=16)

    # Rimuovi etichette pie
    ax_donut.set_xticks([])
    ax_donut.set_yticks([])

    # ----------------------------
    # 6. Nota in basso
    # ----------------------------
    ax_note.text(0.5, 0.5,
                 "Note: Each pass ends at the marker symbol. Team attacks left→right",
                 ha='center', va='center', fontsize=12, color='gray')

    return fig

def plot_pass_network(
    pass_between, 
    average_locations,
    team_name, 
    lineup,
    line_color='#BF616A',
    node_fill='#22312b',
    node_edge='#BF616A',
    pitch_bg='#22312b',
    title='CIao',
    sub_title='',
):
    """
    Draw a pass network on a football pitch in Plotly.
    
    Parameters
    ----------
    pass_between : pd.DataFrame
        Must contain x, y, x_end, y_end, pass_count.
    average_locations : pd.DataFrame
        Indexed by jersey (or player), with x, y, count, top_pass_to, top_pass_from.
    """
    # 1) Create the base pitch
     # Subplot layout: 2 rows, 2 columns
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # title + donut
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        column_widths=[0.75, 0.15],
        row_heights=[0.15, 0.75,0.05],    
        vertical_spacing=0.03
    )

    # === Title and subtitle ===
    fig.add_annotation(
        text=f"<b>{title}</b><br><span style='font-size:14px;color:gray'>{sub_title}</span>",
        xref="paper", yref="paper",
        x=0.1, y=0.975,
        showarrow=False,
        font=dict(size=20),
        align="left"
    )
    fig.update_layout(
        xaxis1=dict(visible=False),
        yaxis1=dict(visible=False)
    )
    # fig = create_pitch_figure()
    
    # === Pitch setup (bottom row) ===
    fig.update_layout(paper_bgcolor=pitch_bg, plot_bgcolor=pitch_bg)
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=2, col=1, autorange="reversed")

    pitch = create_pitch_figure()
    for shape in pitch.layout.shapes:
        fig.add_shape(shape, row=2, col=1)


    # 2) Draw passes as lines, scaled by pass_count
    for _, r in pass_between.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r.x, r.x_end],
                y=[r.y, r.y_end],
                mode='lines',
                line=dict(
                    color=line_color,
                    width=max(r.pass_count, 1)  # ensure at least 1px
                ),
                hoverinfo='none',
                showlegend=False
            ), row=2, col=1
        )

    # 3) Draw nodes at each average location
    for jersey, r in average_locations.iterrows():
        player_name = lineup.loc[
            (lineup['team_name'] == team_name) & (lineup['jersey_number'] == jersey), 'player_name'
        ].iloc[0]

        max_recipient_jersey, max_recipient_pc = pass_between.loc[pass_between.loc[pass_between['passer'] == jersey]['pass_count'].idxmax()][['pass_recipient', 'pass_count']].values
        max_recipient_player_name = lineup.loc[
            (lineup['team_name'] == team_name) & (lineup['jersey_number'] == max_recipient_jersey), 'player_name'
        ].iloc[0]

        max_passer_jersey, max_passer_pc = pass_between.loc[pass_between.loc[pass_between['pass_recipient'] == jersey]['pass_count'].idxmax()][['passer', 'pass_count']].values
        max_passer_player_name = lineup.loc[
            (lineup['team_name'] == team_name) & (lineup['jersey_number'] == max_passer_jersey), 'player_name'
        ].iloc[0]
        
        hover = (
            f"<b>{player_name}</b><br>"
            f"Total passes: {int(r['count'])}<br>"
            f"Most passes to: {max_recipient_player_name} #{int(max_recipient_jersey)} ({int(max_recipient_pc)}) <br>"
            f"Most passes from: {max_passer_player_name} #{int(max_passer_jersey)} ({int(max_passer_pc)})"
        )

        fig.add_trace(
            go.Scatter(
                x=[r.x],
                y=[r.y],
                mode='markers',
                marker=dict(
                   size=r['count'],    # adjust multiplier to taste
                    color=node_fill,
                    line=dict(color=node_edge, width=2)
                ),
                hovertemplate=hover,
                showlegend=False,
                name=''

            ), row=2, col=1
        )

        fig.add_annotation(
            text=jersey,
            x=r.x, y=r.y, xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="black", weight='bold')
        )

        fig.update_layout(
        autosize=False,
        width = 800,
        height=700,
        margin=dict(t=60, b=40, l=20, r=20),)

    # Add annotation note below the pitch
    fig.add_annotation(
        text="Note: Data up to first team substitution | Team attack from left to right",
        xref="paper", yref="paper",
        x=0.5, y=0, xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    return fig