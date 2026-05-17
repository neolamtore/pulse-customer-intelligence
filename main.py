import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

np.random.seed(7)

# -------------------------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------
# COLOUR PALETTE
# -------------------------------------------------
C = {
    "bg": "#0A0A0A",
    "panel": "#111111",
    "yellow": "#E8FF00",
    "coral": "#FF4D4D",
    "blue": "#00CFFF",
    "green": "#00FF94",
    "white": "#F0EDE6",
    "muted": "#555550"
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor": C["panel"],
    "text.color": C["white"],
    "axes.labelcolor": C["muted"],
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "font.family": "monospace"
})

# -------------------------------------------------
# SYNTHETIC DATA
# -------------------------------------------------
products = [
    "AirPods Clone",
    "Smartwatch",
    "USB Dock",
    "Keyboard",
    "Webcam",
    "SSD",
]

N = 3200
start = datetime(2025, 1, 1)

df = pd.DataFrame({
    "product": np.random.choice(products, N),
    "rating": np.random.choice([1,2,3,4,5], N, p=[0.08,0.12,0.15,0.30,0.35]),
    "date": [start + timedelta(days=np.random.randint(0,365)) for _ in range(N)],
    "region": np.random.choice(
        ["London","Manchester","Birmingham","Leeds"],
        N
    ),
    "order_value": np.random.uniform(20,220,N)
})

df["week"] = df["date"].dt.isocalendar().week.astype(int)

# -------------------------------------------------
# PRODUCT METRICS
# -------------------------------------------------
prod_stats = df.groupby("product").agg({
    "rating":"mean",
    "order_value":"sum"
})

prod_stats.columns = ["avg_rating","revenue"]

prod_stats["volatility"] = np.random.uniform(0.15,0.6,len(prod_stats))
prod_stats["loyalty"] = np.random.uniform(0.4,0.9,len(prod_stats))
prod_stats["momentum"] = np.random.uniform(-0.2,0.4,len(prod_stats))

prod_stats["churn_risk"] = (
    (5-prod_stats["avg_rating"])/4*0.5 +
    (1-prod_stats["loyalty"])*0.3 +
    prod_stats["volatility"]*0.2
)

# -------------------------------------------------
# FIGURE
# -------------------------------------------------
fig = plt.figure(figsize=(22,18))

# -------------------------------------------------
# 1 SENTIMENT TREND
# -------------------------------------------------
ax1 = plt.subplot(231)

weekly = df.groupby("week")["rating"].mean()

ax1.plot(
    weekly.index,
    weekly.values,
    color=C["yellow"],
    linewidth=2
)

ax1.set_title("Sentiment Momentum")

# -------------------------------------------------
# 2 CHURN RISK
# -------------------------------------------------
ax2 = plt.subplot(232)

risk = prod_stats["churn_risk"].sort_values()

ax2.barh(
    risk.index,
    risk.values,
    color=C["coral"]
)

ax2.set_title("Churn Risk")

# -------------------------------------------------
# 3 VOLATILITY
# -------------------------------------------------
ax3 = plt.subplot(233)

vol = prod_stats["volatility"]

ax3.bar(
    vol.index,
    vol.values,
    color=C["blue"]
)

ax3.set_title("Sentiment Volatility")
plt.xticks(rotation=30)

# -------------------------------------------------
# 4 HEATMAP
# -------------------------------------------------
ax4 = plt.subplot(234)

heat = df.groupby([
    df["date"].dt.dayofweek,
    "rating"
]).size().unstack(fill_value=0)

smooth = gaussian_filter(heat.values, sigma=1)

cmap = LinearSegmentedColormap.from_list(
    "pulse",
    [C["panel"], C["green"], C["yellow"]]
)

ax4.imshow(smooth, cmap=cmap)

ax4.set_title("Review Density")

# -------------------------------------------------
# 5 REGIONAL RATINGS
# -------------------------------------------------
ax5 = plt.subplot(235)

regional = df.groupby("region")["rating"].mean()

ax5.barh(
    regional.index,
    regional.values,
    color=C["green"]
)

ax5.set_title("Regional Performance")

# -------------------------------------------------
# 6 REVENUE VS RATING
# -------------------------------------------------
ax6 = plt.subplot(236)

ax6.scatter(
    prod_stats["avg_rating"],
    prod_stats["revenue"],
    s=200,
    color=C["yellow"]
)

for p,row in prod_stats.iterrows():
    ax6.text(
        row["avg_rating"],
        row["revenue"],
        p,
        fontsize=8
    )

ax6.set_title("Revenue vs Satisfaction")

# -------------------------------------------------
# SAVE
# -------------------------------------------------
plt.tight_layout()

out = "outputs/pulse_dashboard.png"

plt.savefig(
    out,
    dpi=160,
    facecolor=C["bg"]
)

plt.close()

print(f"saved → {out}")