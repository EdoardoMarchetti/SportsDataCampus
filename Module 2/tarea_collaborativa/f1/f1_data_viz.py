
#Data manipulation libraries
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mplsoccer import FontManager
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from highlight_text import ax_text
import plotly.graph_objects as go 
import pandas as pd
import streamlit as st


from urllib.request import urlopen

fonts_dictionary = {
    'Relaway':{ # per cercare altri font Relaway -> https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/
        'bold':'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-Bold.ttf',
        'regular': 'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-Regular.ttf',
        'italic' : 'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-Italic.ttf',
        'medium' : 'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-Medium.ttf',
        'bold_italic': 'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-BoldItalic.ttf',
        'semibold': 'https://github.com/cyrealtype/Raleway/raw/master/fonts/ttf/Raleway-SemiBold.ttf'
    }
}

COLORS = {
    'oro_sd' : '#C7BE87',
    'grigio_sd' : '#71777f'
}

#Get Font Manager
def get_font_manager(font_name='Relaway', font_weight='regular'):
    """
    Ottiene il gestore del font per un font e un peso specificati.

    Args:
    - font_name (str, opzionale): Nome del font.
    - font_weight (str, opzionale): Peso del font.

    Returns:
    - FontManager: Oggetto gestore del font per il font e il peso specificati.
    """
    return FontManager(fonts_dictionary[font_name][font_weight])

def add_text_to_fig(fig, text, x, y,  color=COLORS['grigio_sd'], ha='center', va='center', fontproperties=get_font_manager().prop, size=9):
    ax = fig.gca()
    ax.text(
        x, y,
        text, 
        color=color, 
        ha=ha, va=va,
        fontproperties=fontproperties,
        size=size,
        transform=ax.transAxes
    )
    return fig

def add_image_to_fig(fig, img_url, 
                     
                     x, y, width=0.5, 
                     height=0.5, background=False, xycoords='axes fraction'):

    image = Image.open(urlopen(img_url)).convert('RGBA')
    
    if background:
        # Add the image as a background for the entire figure
        ax = fig.add_axes([0.275, 0.25, 0.5, 0.5], zorder=0, alpha=0.3)
        ax.imshow(image, extent=[0, 1, 0, 1], zorder=0, alpha=0.1)
        ax.set_axis_off()  # Hide the axis for the background image
    else:
        # Add the image as an overlay at the given (x, y) position
        ax = fig.gca()
        imagebox = OffsetImage(image, zoom=width)  # Zoom determines size based on original resolution
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords=xycoords)
        ax.add_artist(ab)
    return fig

def create_top10_table_image_f1(
    table_df,
    metric_name="points",
    metric_visible_name="Punti",
    image_column="imageDataURL",
    name_column="name",
    team_column="team_name",
    selected_id=None,
    id_column=None,
    team_logo_column=None,
    img_width=0.35
):
    fig, ax = plt.subplots(figsize=(7, 12))
    ax.axis("off")
    renderer = ax.figure.canvas.get_renderer()
    font_prop = get_font_manager(font_name="Relaway", font_weight="regular").prop
    font_prop_bold = get_font_manager(font_name="Relaway", font_weight="bold").prop

    widths = []
    for i, (_, record) in enumerate(table_df.iterrows()):
        y_pos = len(table_df) - i

        # Rank
        ax.text(
            x=0.1,
            y=y_pos,
            s=f"{record['rank']}°",
            ha="center",
            va="center",
            fontproperties=font_prop_bold,
            size=14,
            color=COLORS["grigio_sd"],
        )

        # Main image (driver/team)
        add_image_to_fig(
            fig=fig,
            img_url=record[image_column],
            x=1,
            y=y_pos,
            xycoords="data",
            width=img_width,
        )

        # Optional team logo
        if team_logo_column and pd.notna(record.get(team_logo_column)):
            add_image_to_fig(
                fig=fig,
                img_url=record[team_logo_column],
                x=2,
                y=y_pos,
                xycoords="data",
                width=0.3,
            )

        # Name
        name_text = ax.text(
            x=3,
            y=y_pos + 0.1,
            s=record[name_column],
            ha="left",
            va="center",
            fontproperties=font_prop_bold,
            size=16,
            color=COLORS["oro_sd"],
        )
        widths.append(name_text.get_window_extent(renderer=renderer).width)

        # Team name (if available)
        if team_column in record:
            team_text = ax.text(
                x=3,
                y=y_pos - 0.1,
                s=record[team_column],
                ha="left",
                va="top",
                fontproperties=font_prop,
                size=12,
                color=COLORS["grigio_sd"],
            )
            widths.append(team_text.get_window_extent(renderer=renderer).width)

    max_width_data = max(widths) / ax.figure.dpi * 2.54
    value_x = int(3 + max_width_data)
    xmin, xmax = (-0.5, value_x + 1)

    # Linea iniziale
    ax.plot(
        [-0.5, value_x + 1],
        [len(table_df) + 1 - 0.5, len(table_df) + 1 - 0.5],
        color=COLORS["oro_sd"],
    )

    for i, (_, record) in enumerate(table_df.iterrows()):
        y_pos = len(table_df) - i
        val = record[metric_name] if isinstance(record[metric_name], int) else round(record[metric_name], 2)

        ax.text(
            x=value_x,
            y=y_pos,
            s=val,
            ha="left",
            va="center",
            fontproperties=font_prop_bold,
            size=15,
            color=COLORS["grigio_sd"],
        )

        # Background color
        is_selected = selected_id is not None and record.get(id_column) == selected_id
        patch_color = COLORS["oro_sd"] if is_selected else ("white" if i % 2 == 0 else "lightgray")

        ax.add_patch(
            Rectangle(
                xy=(-0.5, y_pos - 0.5),
                width=value_x + 1.5,
                height=1,
                zorder=-1,
                color=patch_color,
                alpha=0.2,
            )
        )

        ax.plot(
            [xmin, xmax],
            [y_pos - 0.5, y_pos - 0.5],
            color=COLORS["oro_sd"],
        )

    ax.set_ylim(top=len(table_df) + 1)
    ax.set_xlim((xmin, xmax))

    ax_text(
        x=(xmax + xmin) / 2,
        y=len(table_df) + 1,
        s=f"<Top 10 {metric_visible_name}>",
        color="black",
        highlight_textprops=[
            {"fontsize": 20, "color": COLORS["oro_sd"]},
        ],
        ax=ax,
        fontproperties=font_prop,
        ha="center",
        va="center",
        textalign="center",
    )

    plt.tight_layout()
    return fig

def plot_super_time(df: pd.DataFrame, by: str = "driver"):
    """
    Crea il grafico scatter-line per Super Time.
    """
    fig = go.Figure()

    if by == "team":
        # Prendi il giro più veloce per ciascun team e GP
        indexes = df.sort_values("time_ms").drop_duplicates(subset=["entity_name", "race_index"], keep="first").sort_index().index

        df = df.loc[indexes]
    
    for name, group in df.groupby("entity_name"):

        fig.add_trace(go.Scatter(
            x=group["race_index"],
            y=(group["superTimeRatio"] * 100),  # da 100%
            mode="lines+markers",
            name=name,
            text=[  # Tooltip personalizzato
                f"""
                <b>{row["entity_name"]}</b><br>
                <b>GP:</b> {row['race_name']}<br>
                <b>Best Lap:</b> {row['time']}<br>
                <b>Delta%:</b> {row['superTimeDelta%']}%
                """ for _, row in group.iterrows()
            ],
            hoverinfo="text"
        ))

    fig.update_layout(
        title=f"Super Time Ratio - {'Drivers' if by == 'driver' else 'Teams'}",
        xaxis_title="GP",
        yaxis_title="Super Time % (100 = best)",
        yaxis=dict(range=[100, df["superTimeRatio"].max() * 105]),  # per un po’ di margine
        hovermode="closest",
        template="plotly_white",
        height=600
    )
    
    return fig