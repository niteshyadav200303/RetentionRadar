"""Generate figures and 01_eda.ipynb for Phase 1."""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_transactions.csv"
NB_PATH = BASE_DIR / "notebooks" / "01_eda.ipynb"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NB_PATH.parent, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# 1. Monthly revenue trend and monthly active customers (line charts)
monthly = df.groupby("YearMonth").agg(
    Revenue=("Revenue", "sum"),
    ActiveCustomers=("CustomerID", "nunique")
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
color = "#1f77b4"
ax1.set_xlabel("Year-Month", fontweight="bold")
ax1.set_ylabel("Total Monthly Revenue (£)", color=color, fontweight="bold")
ax1.plot(monthly["YearMonth"], monthly["Revenue"], color=color, marker="o", linewidth=2.5, label="Monthly Revenue")
ax1.tick_params(axis="y", labelcolor=color)
ax1.tick_params(axis="x", rotation=45)

ax2 = ax1.twinx()
color = "#ff7f0e"
ax2.set_ylabel("Monthly Active Customers", color=color, fontweight="bold")
ax2.plot(monthly["YearMonth"], monthly["ActiveCustomers"], color=color, marker="s", linestyle="--", linewidth=2.5, label="Active Customers")
ax2.tick_params(axis="y", labelcolor=color)
ax2.grid(False)

plt.title("Monthly Revenue Trend & Active Customers (2009-2011)", fontsize=14, fontweight="bold", pad=15)
plt.savefig(FIG_DIR / "monthly_revenue_and_active_customers.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Orders-per-customer distribution (log scale histogram)
orders_per_cust = df.groupby("CustomerID")["Invoice"].nunique()

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(orders_per_cust, log_scale=True, bins=35, kde=True, color="#2ca02c", ax=ax)
ax.set_title("Distribution of Orders per Customer (Log Scale)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Number of Orders (Log Scale)", fontweight="bold")
ax.set_ylabel("Customer Count", fontweight="bold")
plt.savefig(FIG_DIR / "orders_per_customer_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Top 10 countries by revenue (bar)
top_countries = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=top_countries, x="Revenue", y="Country", palette="Blues_r", ax=ax)
ax.set_title("Top 10 Countries by Total Revenue", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Revenue (£)", fontweight="bold")
ax.set_ylabel("Country", fontweight="bold")
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"£{width/1e6:.2f}M", (width, p.get_y() + p.get_height() / 2.),
                ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=9)
plt.savefig(FIG_DIR / "top_10_countries_by_revenue.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. RFM distributions and heatmap of correlations
max_date = df["InvoiceDate"].max()
cutoff_date = max_date - pd.Timedelta(days=90)

obs_df = df[df["InvoiceDate"] < cutoff_date]
perf_df = df[df["InvoiceDate"] >= cutoff_date]

rfm = obs_df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda d: (cutoff_date - d.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(rfm[["Recency", "Frequency", "Monetary"]].corr(), annot=True, cmap="coolwarm", fmt=".3f", ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title("Correlation Heatmap (RFM Features)", fontweight="bold")

sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Recency", palette="viridis", alpha=0.7, ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_xscale("log")
axes[1].set_title("RFM Scatter (Log Frequency vs Log Monetary)", fontweight="bold")
plt.savefig(FIG_DIR / "rfm_correlations.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. Churn rate overview
obs_custs = set(obs_df["CustomerID"].unique())
perf_custs = set(perf_df["CustomerID"].unique())
churn_status = pd.Series([1 if c not in perf_custs else 0 for c in obs_custs])
churn_counts = churn_status.value_counts()

fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Churned (1)", "Retained (0)"]
values = [churn_counts[1], churn_counts[0]]
colors = ["#e74c3c", "#2ecc71"]
bars = ax.bar(labels, values, color=colors, width=0.5)
ax.set_title(f"Customer Churn Distribution (90-Day Post Cutoff)\nChurn Rate: {churn_counts[1]/len(obs_custs):.1%}", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers", fontweight="bold")
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2., h + 50, f"{h} ({h/len(obs_custs):.1%})", ha="center", va="bottom", fontweight="bold")
plt.savefig(FIG_DIR / "churn_rate_overview.png", dpi=300, bbox_inches="tight")
plt.close()

print("All charts generated and saved to reports/figures/.")

# Create Jupyter Notebook
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Exploratory Data Analysis (EDA)\n",
                "## E-Commerce Customer Churn & Lifetime Value (LTV) Predictor\n",
                "This notebook provides key data cleaning verification, transaction dynamics, country distribution, RFM profiles, and the target churn rate overview."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "DATA_PATH = '../data/processed/clean_transactions.csv'\n",
                "df = pd.read_csv(DATA_PATH)\n",
                "df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\n",
                "print('Loaded clean transactions shape:', df.shape)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Dataset Shape & Missing Values Before / After Cleaning"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "summary_table = pd.DataFrame({\n",
                "    'Stage': ['Raw UCI Online Retail II', 'Cleaned Transactions'],\n",
                "    'Rows': [1067371, len(df)],\n",
                "    'Missing CustomerIDs': [243007, 0],\n",
                "    'Cancelled Invoices': [19494, 0],\n",
                "    'Non-positive Qty/Price': [22950, 0]\n",
                "})\n",
                "summary_table"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Monthly Revenue Trend & Monthly Active Customers\n",
                "Examining seasonal sales patterns and active customer trends over the 2-year lifespan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)\n",
                "monthly = df.groupby('YearMonth').agg(Revenue=('Revenue', 'sum'), ActiveCustomers=('CustomerID', 'nunique')).reset_index()\n",
                "\n",
                "fig, ax1 = plt.subplots(figsize=(12, 5))\n",
                "color = '#1f77b4'\n",
                "ax1.plot(monthly['YearMonth'], monthly['Revenue'], color=color, marker='o', linewidth=2.5, label='Monthly Revenue')\n",
                "ax1.set_xlabel('Year-Month', fontweight='bold')\n",
                "ax1.set_ylabel('Total Revenue (£)', color=color, fontweight='bold')\n",
                "ax1.tick_params(axis='y', labelcolor=color)\n",
                "ax1.tick_params(axis='x', rotation=45)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "color = '#ff7f0e'\n",
                "ax2.plot(monthly['YearMonth'], monthly['ActiveCustomers'], color=color, marker='s', linestyle='--', linewidth=2.5, label='Active Customers')\n",
                "ax2.set_ylabel('Active Customers', color=color, fontweight='bold')\n",
                "ax2.tick_params(axis='y', labelcolor=color)\n",
                "ax2.grid(False)\n",
                "plt.title('Monthly Revenue Trend & Active Customers (2009-2011)', fontsize=14, fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Orders per Customer Distribution (Log Scale)\n",
                "Checking repeat purchasing behavior across the customer base."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "orders_per_cust = df.groupby('CustomerID')['Invoice'].nunique()\n",
                "fig, ax = plt.subplots(figsize=(8, 4))\n",
                "sns.histplot(orders_per_cust, log_scale=True, bins=35, kde=True, color='#2ca02c', ax=ax)\n",
                "ax.set_title('Orders per Customer (Log Scale)', fontsize=13, fontweight='bold')\n",
                "ax.set_xlabel('Number of Orders (Log Scale)', fontweight='bold')\n",
                "ax.set_ylabel('Customer Count', fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Top 10 Countries by Revenue\n",
                "Evaluating geographic market concentration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "top_countries = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()\n",
                "fig, ax = plt.subplots(figsize=(9, 4.5))\n",
                "sns.barplot(data=top_countries, x='Revenue', y='Country', palette='Blues_r', ax=ax)\n",
                "ax.set_title('Top 10 Countries by Total Revenue', fontsize=13, fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. RFM Distributions and Correlation Heatmap"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "max_date = df['InvoiceDate'].max()\n",
                "cutoff_date = max_date - pd.Timedelta(days=90)\n",
                "obs_df = df[df['InvoiceDate'] < cutoff_date]\n",
                "rfm = obs_df.groupby('CustomerID').agg(\n",
                "    Recency=('InvoiceDate', lambda d: (cutoff_date - d.max()).days),\n",
                "    Frequency=('Invoice', 'nunique'),\n",
                "    Monetary=('Revenue', 'sum')\n",
                ").reset_index()\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n",
                "sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', fmt='.3f', ax=axes[0], vmin=-1, vmax=1)\n",
                "axes[0].set_title('RFM Feature Correlation', fontweight='bold')\n",
                "sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Recency', palette='viridis', alpha=0.7, ax=axes[1])\n",
                "axes[1].set_yscale('log')\n",
                "axes[1].set_xscale('log')\n",
                "axes[1].set_title('RFM Scatter (Log Scale)', fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Churn Rate Overview (Phase 2 Target Definition)\n",
                "Cutoff = Last InvoiceDate minus 90 days. Churn = 1 if no purchases made in the 90 days after cutoff."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [],
            "source": [
                "perf_df = df[df['InvoiceDate'] >= cutoff_date]\n",
                "obs_custs = set(obs_df['CustomerID'].unique())\n",
                "perf_custs = set(perf_df['CustomerID'].unique())\n",
                "\n",
                "churn_labels = pd.Series([1 if cid not in perf_custs else 0 for cid in obs_custs])\n",
                "counts = churn_labels.value_counts()\n",
                "print(f'Total Observed Customers: {len(obs_custs)}')\n",
                "print(f'Churned: {counts[1]} ({counts[1]/len(obs_custs):.2%})')\n",
                "print(f'Retained: {counts[0]} ({counts[0]/len(obs_custs):.2%})')\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(6, 4))\n",
                "ax.bar(['Churned (1)', 'Retained (0)'], [counts[1], counts[0]], color=['#e74c3c', '#2ecc71'], width=0.5)\n",
                "ax.set_title('Customer Churn Distribution', fontweight='bold')\n",
                "ax.set_ylabel('Customer Count', fontweight='bold')\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Jupyter notebook saved to {NB_PATH}")
