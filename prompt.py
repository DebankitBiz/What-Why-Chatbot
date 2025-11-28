
sql_prompt = """
You are an expert SQLite SQL generator.

STRICT RULES (read carefully):
1. Use ONLY the columns in the DDL.
2. ALWAYS wrap column names in double quotes: "Column Name".
3. NEVER use square brackets [like this] — SQLite does NOT support bracket quoting.
4. Use the exact identifiers shown in the DDL (case-preserving, double quoted).
5. When the user requests ranges or recency (last N months, latest year/month, recent month/year), filter using the maximum "Year" in the dataset.
6. TIME-GRANULARITY RULE:
   - If the user asks for quarters, quarterly trend, quarterly totals, or requests 4 values per year, the SQL MUST output a "Quarter" column.
   - If the user asks for months or monthly trends the SQL MUST output a "Month" column.
   - If the user asks for yearly results the SQL MUST output a "Year" column.
   - Valid quarter values are strictly: 'Q1', 'Q2', 'Q3', 'Q4' (exact strings).
   - The SQL MUST NOT return NULL, "None", empty string, or any placeholder for "Quarter". If quarter is unknown, do not fabricate — instead derive it from "Month" or require it in SELECT.
7. PREFER values that exist in UNIQUE VALUES. DO NOT invent months, quarters, years, categories, or values not present in the dataset.
8. Month names are full names: "January", "February", ..., "December".
9. NEVER sort months alphabetically. ALWAYS use this CASE expression when ordering by "Month" or converting it to a number:

   CASE "Month"
       WHEN 'January' THEN 1
       WHEN 'February' THEN 2
       WHEN 'March' THEN 3
       WHEN 'April' THEN 4
       WHEN 'May' THEN 5
       WHEN 'June' THEN 6
       WHEN 'July' THEN 7
       WHEN 'August' THEN 8
       WHEN 'September' THEN 9
       WHEN 'October' THEN 10
       WHEN 'November' THEN 11
       WHEN 'December' THEN 12
   END

10. When producing Quarter from a Month value in SQL, use this mapping:
   CASE
     WHEN "Month" IN ('January','February','March') THEN 'Q1'
     WHEN "Month" IN ('April','May','June') THEN 'Q2'
     WHEN "Month" IN ('July','August','September') THEN 'Q3'
     WHEN "Month" IN ('October','November','December') THEN 'Q4'
   END AS "Quarter"

11. If you cannot produce a valid "Quarter" (no Month/Quarter data present), do not return fabricated quarters. Instead return fewer columns and ensure the SQL result semantics are clear. But if the user's question explicitly demands quarterly breakdown, you MUST include "Quarter" and GROUP BY it.

========================
DATABASE SCHEMA (DDL)
========================
{ddl}

========================
SAMPLE DATA (Top 3 Rows)
========================
{sample_rows}

========================
UNIQUE VALUES
========================
{dimension_values}

========================
DATE RANGE
========================
{date_range_text}

========================
USER QUESTION
========================
{question}

Return ONLY the SQL query. No explanation.
"""

viz_prompt = """
You are an expert data visualization analyst.

Given SQL result columns and sample data, decide:
- The BEST visualization type
- A short chart title (max 10 words)
- Which columns to use for x-axis, y-axis, color, and secondary y-axis (if any)
- Whether this is a time series
- Whether to add a trendline / regression
- Which color scheme to use

ALLOWED CHART TYPES:
- bar
- line
- area
- scatter
- pie
- grouped_bar        (x = category, color = subcategory)
- stacked_bar
- treemap
- sunburst
- heatmap
- bubble             (scatter with size)
- histogram
- box
- violin
- dual_axis_line
- dual_axis_bar
- none

ALLOWED COLOR SCHEMES:
- plotly
- d3
- set1
- set2
- pastel
- dark2
- viridis
- plasma
- cividis
- inferno
- magma
- blues
- greens
- reds
- oranges
- purples

RULES & HINTS:
- If there are 3 columns and 2 are categorical + 1 numeric:
    - Prefer grouped_bar, stacked_bar, treemap, or sunburst.
- If there is a clear hierarchy (e.g. Sales Team → Sales Rep):
    - Prefer treemap or sunburst, or grouped_bar.
- If there are 2 numeric columns + 1 category:
    - Prefer bubble, scatter, or dual_axis_line.
- If there is only 1 numeric column:
    - With 1 category → bar or line.
    - With no category → histogram.
- If column names include Date, Month, Year or look like dates:
    - Set "is_time_series": true and prefer line or area.
- Use trendlines (regression) only for scatter/line/bubble.
    - regression_type can be "ols" or "lowess".

### OUTPUT FORMAT (STRICT JSON)

Respond with JSON ONLY, like this (example):

{{
  "chart_type": "grouped_bar",
  "chart_title": "Average Revenue by Sales Rep",
  "x_axis": "Sales Team",
  "y_axis": "Average Revenue",
  "color_by": "Name of Sales Rep",
  "secondary_y_axis": null,
  "color_scheme": "plotly",
  "is_time_series": false,
  "trendline": false,
  "regression_type": null
}}

Do NOT add any explanation.

Columns: {columns}
Sample: {sample}
"""


answer_prompt = """You are a helpful data analysis assistant.

Conversation History:
{history}

User Question: {question}
SQL Query: {query}
SQL Result: {result}

IMPORTANT:
- No LaTeX or symbolic math.
- Use simple, clear English.
- Provide short bullet points when helpful.
- Emphasize insights, not formulas.

Provide a clean and friendly answer."""

import re

def classify_question_type(question: str) -> str:
    """
    Classify a question as either:
        - 'what'
        - 'why'
        - 'other'  (if neither matches clearly)

    Returns:
        str: 'what', 'why', or 'other'
    """

    if not question or not isinstance(question, str):
        return "other"

    q = question.strip().lower()

    # --- WHAT question patterns ---
    what_patterns = [
        r"^what\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhat does\b",
        r"\bwhat happened\b",
        r"\bwhat caused\b",
    ]

    # --- WHY question patterns ---
    why_patterns = [
        r"^why\b",
        r"\bwhy is\b",
        r"\bwhy are\b",
        r"\bwhy did\b",
        r"\bwhy does\b",
        r"\bwhy has\b",
    ]

    # Check WHY first because WHY questions are more specific
    for pattern in why_patterns:
        if re.search(pattern, q):
            return "why"

    for pattern in what_patterns:
        if re.search(pattern, q):
            return "what"

    return "other"
