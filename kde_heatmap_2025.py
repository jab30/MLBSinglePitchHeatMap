import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from api_scraper import MLB_Scrape

# --- Load App ---
st.title("Pitch Location KDE Heatmap (2025)")

# --- Scraper & Data ---
scraper = MLB_Scrape()
players_df = scraper.get_players(season=2025, sport_id=1, game_type=["R"])

# --- Extract player names and sort ---
player_names = players_df.select("name").to_series().to_list()
player_names = sorted(player_names)

# --- Sidebar Inputs ---
pitcher_name_input = st.selectbox(
    "Select Pitcher",
    player_names,
    index=player_names.index("Eury Pérez") if "Eury Pérez" in player_names else 0
)
pitch_type_input = st.selectbox("Pitch Type", ["FF", "SL", "CH", "CU", "SI", "FC"])
batter_hand_input = st.selectbox("Batter Handedness", ["Both", "Left", "Right"])
start_date = st.date_input("Start Date", pd.to_datetime("2025-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-12-31"))

# --- Match pitcher name to ID ---
matching = players_df.filter(pl.col("name") == pitcher_name_input)
if matching.is_empty():
    st.error("Pitcher name not found in player list.")
    st.stop()

player_id = matching.select("player_id").to_series().to_list()[0]
game_ids = scraper.get_player_games_list(player_id=player_id, season=2025, pitching=True)
raw_data = scraper.get_data(game_ids)
df = scraper.get_data_df(raw_data)

# --- Filter by pitcher name and pitch type ---
df = df.filter(pl.col("pitcher_name") == pitcher_name_input)
df = df.filter(pl.col("pitch_type") == pitch_type_input)
df = df.filter((pl.col("game_date") >= str(start_date)) & (pl.col("game_date") <= str(end_date)))

# --- Batter Handedness Filter ---
if batter_hand_input == "Left":
    df = df.filter(pl.col("batter_hand") == "L")
elif batter_hand_input == "Right":
    df = df.filter(pl.col("batter_hand") == "R")


# --- If no data after filter ---
if df.is_empty():
    st.warning("No matching data found for that pitcher, pitch type, and date range.")
    st.stop()

# --- VAA Calculations ---
y0 = 50
yf = 17 / 12
df = df.with_columns([
    (-((pl.col("vy0") ** 2) - (2 * pl.col("ay") * (y0 - yf))).sqrt()).alias("vy_f")
]).with_columns([
    ((pl.col("vy_f") - pl.col("vy0")) / pl.col("ay")).alias("t")
]).with_columns([
    (pl.col("vz0") + (pl.col("az") * pl.col("t"))).alias("vz_f")
]).with_columns([
    (-pl.col("vz_f") / pl.col("vy_f")).arctan().alias("vaa_rad")
]).with_columns([
    (pl.col("vaa_rad") * (180 / np.pi)).alias("vaa_deg")
])

# --- Summary Stats ---
summary = df.select([
    pl.col("start_speed").mean().round(1).alias("Velo"),
    pl.col("ivb").mean().round(1).alias("IVB"),
    pl.col("hb").mean().round(1).alias("HB"),
    pl.col("extension").mean().round(1).alias("Ext"),
    pl.col("vaa_deg").mean().round(1).alias("VAA")
]).to_pandas().iloc[0]

# --- Headshot Function ---
def player_headshot(ax: plt.Axes):
    try:
        pitcher_id = df.select("pitcher_id").unique().to_series().to_list()[0]
        url = f"https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pitcher_id}/headshot/silo/current.png"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        imagebox = OffsetImage(img, zoom=0.30)
        ab = AnnotationBbox(imagebox, (1.3, 0.85), frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print("Headshot load failed:", e)

# --- Prepare KDE Data ---
pdf = df.select(["px", "pz"]).drop_nulls().to_pandas()

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(
    data=pdf,
    x="px",
    y="pz",
    fill=True,
    cmap="OrRd",
    bw_adjust=0.9,   # tighter kernel
    thresh=0.5,      # show lighter densities
    levels=70,       # smoother color transitions
    ax=ax
)

# Strike zone
strike_zone = plt.Rectangle(
    (-0.83, 1.5), 1.66, 2.0,
    linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--", zorder=10
)
ax.add_patch(strike_zone)

# Summary Text
summary_text = (
    f"Velo: {summary['Velo']} mph\n"
    f"IVB: {summary['IVB']} in\n"
    f"HB: {summary['HB']} in\n"
    f"Ext: {summary['Ext']} ft\n"
    f"VAA: {summary['VAA']}\u00b0"
)
ax.text(-1.9, 0.2, summary_text, fontsize=10, verticalalignment='bottom',
        horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

# Add headshot
player_headshot(ax)

# Finalize
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xlabel("Horizontal Location (ft)")
ax.set_ylabel("Vertical Location (ft)")
ax.set_title(f"{pitcher_name_input} - {pitch_type_input} KDE Heatmap (Catcher POV)")
ax.grid(True, alpha=0.2)
st.pyplot(fig)