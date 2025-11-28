import pandas as pd
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import plotly.graph_objects as go

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def normalize_freq(freq: str) -> str:
    f = str(freq).strip().lower()
    if f in ["w", "week", "weekly"]:
        return "W"
    if f in ["m", "month", "monthly"]:
        return "M"
    if f in ["q", "quarter", "quarterly"]:
        return "Q"
    raise ValueError(f"Unsupported frequency: {freq}. Use weekly/monthly/quarterly.")

def add_period_start(df: pd.DataFrame, date_col: str, freq_code: str,
                     period_col: str = "period_start") -> pd.DataFrame:

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    if freq_code == "W":
        out[period_col] = out[date_col] - pd.to_timedelta(out[date_col].dt.dayofweek, unit="D")
    elif freq_code == "M":
        out[period_col] = out[date_col].dt.to_period("M").dt.to_timestamp()
    elif freq_code == "Q":
        out[period_col] = out[date_col].dt.to_period("Q").dt.to_timestamp()

    return out


def normalize_freq(freq: str) -> str:
    f = str(freq).strip().lower()
    if f in ["w", "week", "weekly"]:
        return "W"
    if f in ["m", "month", "monthly"]:
        return "M"
    if f in ["q", "quarter", "quarterly"]:
        return "Q"
    raise ValueError(f"Unsupported frequency: {freq}. Use weekly/monthly/quarterly.")

def add_period_start(df: pd.DataFrame, date_col: str, freq_code: str,
                     period_col: str = "period_start") -> pd.DataFrame:

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    if freq_code == "W":
        out[period_col] = out[date_col] - pd.to_timedelta(out[date_col].dt.dayofweek, unit="D")
    elif freq_code == "M":
        out[period_col] = out[date_col].dt.to_period("M").dt.to_timestamp()
    elif freq_code == "Q":
        out[period_col] = out[date_col].dt.to_period("Q").dt.to_timestamp()

    return out


def build_contribution_table(df: pd.DataFrame,
                             metric_col: str,
                             period_col: str,
                             dim_cols,
                             min_pct_threshold: float = 0.2):

    if isinstance(dim_cols, str):
        dim_cols = [dim_cols]

    group_cols = [period_col] + dim_cols

    temp = df[group_cols + [metric_col]].copy()
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors='coerce').fillna(0)

    grouped = temp.groupby(group_cols, as_index=False)[metric_col].sum()

    total = (
        grouped.groupby(period_col, as_index=False)[metric_col]
               .sum()
               .rename(columns={metric_col: f"{metric_col}_total"})
    )

    merged = grouped.merge(total, on=period_col, how="left")
    merged["pct"] = merged[metric_col] / merged[f"{metric_col}_total"] * 100

    # Filter small contributors using historical average %
    global_avg = merged.groupby(dim_cols)["pct"].mean().reset_index()
    keep_dims = global_avg[global_avg["pct"] >= min_pct_threshold][dim_cols]

    if keep_dims.empty:
        return merged

    return merged.merge(keep_dims, on=dim_cols, how="inner")


def rca_for_dimension_table(contrib_df: pd.DataFrame,
                            dim_cols,
                            period_col: str,
                            metric_col: str,
                            spike_period,
                            failure_type: str = "higher",
                            history_periods: int = 6,
                            top_n: int = 5):

    if isinstance(dim_cols, str):
        dim_cols = [dim_cols]

    df = contrib_df.copy()
    df = df[df[period_col] <= spike_period]

    all_periods = sorted(df[period_col].unique())
    if spike_period not in all_periods:
        return pd.DataFrame()

    spike_idx = all_periods.index(spike_period)
    start_idx = max(0, spike_idx - history_periods)
    history_list = all_periods[start_idx:spike_idx]

    recent_df = df[df[period_col] == spike_period]
    history_df = df[df[period_col].isin(history_list)]

    if recent_df.empty or history_df.empty:
        return pd.DataFrame()

    history_pct = history_df.groupby(dim_cols)["pct"].mean().reset_index()
    history_pct.rename(columns={"pct": "pct_history"}, inplace=True)

    recent_pct = recent_df[dim_cols + ["pct"]]
    recent_pct.rename(columns={"pct": "pct_recent"}, inplace=True)

    final = recent_pct.merge(history_pct, on=dim_cols, how="left")
    final["pct_history"].fillna(0, inplace=True)

    final["abs_change_pct"] = final["pct_recent"] - final["pct_history"]

    # 🚨 Directional filtering (IMPORTANT)
    if failure_type.lower() == "higher":
        final = final[final["abs_change_pct"] > 0]
        sort_order = False
    else:
        final = final[final["abs_change_pct"] < 0]
        sort_order = True

    # If no rows remain after direction-filter, return empty
    if final.empty:
        return pd.DataFrame()

    final = final.sort_values("abs_change_pct", ascending=sort_order)

    return final[dim_cols + ["pct_recent", "pct_history", "abs_change_pct"]].head(top_n)


def advanced_rca(spike_date,
                 freq: str,
                 failure_type: str = "higher",
                 history_periods: int | None = None,
                 top_n: int = 5):

    dim_1d = ["Product Name", "Region", "Product Class", "Sales Team"]

    dim_2d = [
        ["Product Name", "Region"],
        ["Region", "Sales Team"],
        ["Product Name", "Sales Team"]
    ]

    metric_col = "Sales"
    #df = pd.read_excel("WMQ.xlsx")

    freq_code = normalize_freq(freq)
    spike_ts = pd.to_datetime(spike_date)

    if history_periods is None:
        history_periods = {"W": 8, "M": 6, "Q": 4}.get(freq_code, 6)

    df_with_period = add_period_start(df, "ds", freq_code, "period_start")

    spike_period = add_period_start(
        pd.DataFrame({"ds": [spike_ts]}),
        "ds", freq_code, "period_start"
    )["period_start"].iloc[0]

    results = {}

    # 1D RCA
    for dim in dim_1d:
        contrib = build_contribution_table(df_with_period, metric_col, "period_start",
                                           dim, min_pct_threshold=0.5)

        rca = rca_for_dimension_table(
            contrib, dim, "period_start", metric_col,
            spike_period, failure_type, history_periods, top_n
        )
        results[dim] = rca

    # 2D RCA
    for dims in dim_2d:
        contrib = build_contribution_table(df_with_period, metric_col, "period_start",
                                           dims, min_pct_threshold=0.2)

        rca = rca_for_dimension_table(
            contrib, dims, "period_start", metric_col,
            spike_period, failure_type, history_periods, top_n
        )
        results[" + ".join(dims)] = rca

    return results


def rca_input(user_question):
    prompt = """
You generate ONLY JSON.
Format:
{
  "spike_date": "YYYY-MM-DD",
  "freq": "weekly/monthly/quarterly/daily",
  "failure_type": "higher/lower"
}
NO text. NO markdown. NO explanation.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question}
        ]
    )

    raw = response.output_text.strip()

    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON:\n{raw}")

    return json.loads(match.group(0))


def run_rca(user_question):
    params = rca_input(user_question)

    rca_output = advanced_rca(
        spike_date=params["spike_date"],
        freq=params["freq"],
        failure_type=params["failure_type"]
    )

    return rca_output, params["failure_type"]


def rca_agent(user_question, rca_text, failure_type):
    
    direction_text = "increase" if failure_type.lower() == "higher" else "decline"

    planner_prompt = f"""
You are a Senior Data Scientist and Executive Insight Writer.

Your task is to convert multi-dimensional RCA tables into a polished,
consulting-grade business explanation. Your output MUST follow the rules below.

====================================================
### 1. DIRECTIONAL FILTERING (MANDATORY)

You must strictly follow this logic:

If failure_type = "higher" (spike):
    - USE ONLY rows where abs_change_pct > 0.
    - Never treat negative values as contributors.
    - Positive contributors = Core Drivers.
    - Negative values appear ONLY under "Offsetting Factors".

If failure_type = "lower" (decline):
    - USE ONLY rows where abs_change_pct < 0.
    - Never treat positive values as contributors.
    - Negative contributors = Core Drivers.
    - Positive values appear ONLY under "Offsetting Factors".

Never mix directions. Never show opposite-direction contributors as drivers.

====================================================
### 2. STRUCTURE (MANDATORY)

# Executive Summary
2–3 crisp sentences explaining WHY the {direction_text} happened.

## 1. Primary Driver — <Dimension Name>
List 1–3 strongest contributors (correct direction only).
Format:
- <Item> changed by ±X.XX%

## 2. Secondary Drivers
### <Dimension Name>
- <Item> changed by ±X.XX%

### <Dimension Name>
- <Item> changed by ±X.XX%

(Include only dimensions that have contributors in the correct direction.)

## 3. Interaction Effects
Use readable English names:
- "Product and Sales Team Interaction"
- "Product and Region Interaction"
- "Region and Sales Team Interaction"

Format:
- <Item combination> changed by ±X.XX%

Include only combinations with abs_change_pct >= 0.50%.



====================================================
### 3. FORMATTING RULES

- Follow Markdown hierarchy EXACTLY.
- One bullet per line. No merged lines.
- Percent values MUST have 2 decimals (e.g., -3.45%).
- No raw numbers. No tables.
- No mention of pct_recent or pct_history.
- Dimension names must be readable (no +, no ×).
    Convert "Product Name + Sales Team" → "Product and Sales Team Interaction".

====================================================
Here are the RCA tables (already pre-filtered by direction):
{rca_text}

Now generate the final RCA explanation.
"""

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    return response.output_text.strip()



user_question = "why is there a down in sales in march 2025?"

# input  = rca_input(user_question)
# st.write(input)

# rca_text, failure_type = run_rca(user_question)
# st.write(rca_text)

# for dim, tbl in rca_text.items():
#     st.subheader(f"RCA – {dim}")
#     st.dataframe(tbl)

# summary = rca_agent(user_question, rca_text,failure_type)

# st.markdown("## RCA Explanation")
# st.markdown(summary)


# st.markdown("## 📊 Combined RCA Summary Chart")
# combined_fig = plot_combined_rca(rca_text)
# st.plotly_chart(combined_fig, use_container_width=True)



