def app():
    import os
    import re
    import json
    import sqlite3
    import pandas as pd
    import streamlit as st
    from dotenv import load_dotenv

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from langchain_community.utilities import SQLDatabase
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
    import random

    # ---------------------------------------
    # Setup
    # ---------------------------------------
    load_dotenv()
    st.set_page_config(page_title="Structured Data Assistant", page_icon="📊", layout="wide")

    st.markdown(
        "<h1 style='text-align: center;'>📊 Structured Data Assistant Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 16px;'>Upload an Excel file and chat with your data in natural language.</p>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------
    # Sticky footer CSS (chat input fixed at bottom)
    # ---------------------------------------
    st.markdown(
    """
    <style>
    .chat-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 15px 20px 18px 20px;
        background-color: #ffffff;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
        z-index: 9999 !important;
        border-top: 1px solid #e6e6e6;
    }

    .chat-footer-inner {
        max-width: 900px;
        margin: 0 auto;
    }

    /* Remove extra margin Streamlit adds around input */
    .stTextInput > div > div {
        margin-bottom: 0px !important;
    }

    /* Suggested buttons pinned above input */
    .suggested-button {
        width: 100%;
        text-align: left;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #dcdcdc;
        background: #fafafa;
        font-size: 14px;
        cursor: pointer;
    }

    .suggested-box-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    # ---------------------------------------
    # Chat History
    # ---------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # This will hold a message just submitted (typed or clicked)
    pending_msg = st.session_state.get("pending_user_msg", None)

    # ---------------------------------------
    # Helper: Ensure string for LLM prompt
    # ---------------------------------------
    def ensure_str(x):
        """Ensures all prompt inputs are converted to safe strings."""
        if isinstance(x, str):
            return x
        return json.dumps(x, indent=2, default=str)

    # ---------------------------------------
    # Helper: Clean SQL
    # ---------------------------------------
    def clean_sql(s: str) -> str:
        if not s:
            return ""

        # Extract content inside ```sql ... ``` if present
        m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)

        # Remove "SQL query:" or similar prefixes
        s = re.sub(r"^\s*(sql\s*query|sqlquery)\s*:\s*", "", s, flags=re.IGNORECASE)

        # Trim lines before the first SELECT/WITH
        lines = s.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("select", "with")):
                s = "\n".join(lines[i:])
                break

        # Remove stray backticks and trailing semicolons
        s = s.replace("```", "").strip().rstrip(";")
        return s

    # ---------------------------------------
    # Helper: Fix month ordering in SQL
    # ---------------------------------------
    def fix_month_order(sql: str) -> str:
        month_case = """
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
        """
        sql = re.sub(
            r'ORDER BY\s+"Month"\s*(ASC|DESC)?',
            rf'ORDER BY {month_case} \1',
            sql,
            flags=re.IGNORECASE,
        )
        return sql

    # ---------------------------------------
    # Rephraser (NL summary)
    # ---------------------------------------
    def build_rephraser(llm: ChatOpenAI):
        return (
            PromptTemplate.from_template(
                """You are a helpful data analysis assistant.

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
            )
            | llm
            | StrOutputParser()
        )

    # ---------------------------------------
    # Color palettes + helper
    # ---------------------------------------
    COLOR_QUAL = {
        "plotly": px.colors.qualitative.Plotly,
        "d3": px.colors.qualitative.D3,
        "set1": px.colors.qualitative.Set1,
        "set2": px.colors.qualitative.Set2,
        "pastel": px.colors.qualitative.Pastel,
        "dark2": px.colors.qualitative.Dark2,
    }

    COLOR_CONT = {
        "viridis": px.colors.sequential.Viridis,
        "plasma": px.colors.sequential.Plasma,
        "cividis": px.colors.sequential.Cividis,
        "inferno": px.colors.sequential.Inferno,
        "magma": px.colors.sequential.Magma,
        "blues": px.colors.sequential.Blues,
        "greens": px.colors.sequential.Greens,
        "reds": px.colors.sequential.Reds,
        "oranges": px.colors.sequential.Oranges,
        "purples": px.colors.sequential.Purples,
    }

    def safe_col(suggested, df, fallback_first=True):
        """Return a valid column name based on LLM suggestion or fallbacks."""
        if isinstance(suggested, str) and suggested in df.columns:
            return suggested
        if fallback_first and len(df.columns) > 0:
            return df.columns[0]
        return None

    # ---------------------------------------
    # Combined Chart Metadata Generator (type, title, axes, colors, etc.)
    # ---------------------------------------
    def build_chart_metadata_generator(llm: ChatOpenAI):
        return (
            PromptTemplate.from_template(
                """
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
            )
            | llm
            | StrOutputParser()
        )

    # ---------------------------------------
    # Sidebar Settings
    # ---------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        excel_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
        sheet_name = st.text_input("Sheet Name (optional)", "")
        table_name = "excel_data"

        model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

        api_key = st.text_input("OPENAI_API_KEY", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.divider()
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            if "pending_user_msg" in st.session_state:
                del st.session_state["pending_user_msg"]
            st.rerun()

    # ---------------------------------------
    # Render previous Chat History (above input)
    # ---------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---------------------------------------
    # If there is a new message to process (typed or clicked)
    # ---------------------------------------
    if pending_msg is not None:
        user_msg = pending_msg

        # Show the new user message
        with st.chat_message("user"):
            st.write(user_msg)
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        # If no Excel file, show error and stop
        if excel_file is None:
            with st.chat_message("assistant"):
                st.error("Upload an Excel file first.")
            # Clear pending flag
            del st.session_state["pending_user_msg"]
            # Add error message to history if you want (optional)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Upload an Excel file first."}
            )
            return

        # ---------------------------------------
        # Load Excel → SQLite
        # ---------------------------------------
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name if sheet_name else 0)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error reading Excel: {e}")
            del st.session_state["pending_user_msg"]
            return

        db_path = "excel_chat.db"
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        # ---------------------------------------
        # Metadata Extraction
        # ---------------------------------------
        def generate_ddl(df_in):
            cols = []
            for col, dtype in df_in.dtypes.items():
                if "int" in str(dtype):
                    sql_type = "INTEGER"
                elif "float" in str(dtype):
                    sql_type = "FLOAT"
                else:
                    sql_type = "TEXT"
                cols.append(f'"{col}" {sql_type}')
            return "CREATE TABLE excel_data (\n  " + ",\n  ".join(cols) + "\n);"

        ddl_text = generate_ddl(df)
        sample_rows = ensure_str(df.head(3).to_dict(orient="records"))

        # Dimension values for key columns (safe checks)
        dimension_values = {}
        important_columns = [
            "Customer Name", "Channel", "Sub-channel", "Product Name", "Product Class",
            "Quantity", "Price", "Sales", "Month", "Year",
            "Name of Sales Rep", "Manager", "Sales Team"
        ]
        for col in important_columns:
            if col in df.columns and df[col].dtype == object:
                vals = df[col].dropna().unique().tolist()
                dimension_values[col] = vals[:20]
        dimension_values_text = ensure_str(dimension_values)

        # Month → number mapping
        MONTH_MAP = {
            "January": 1, "February": 2, "March": 3,
            "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9,
            "October": 10, "November": 11, "December": 12
        }

        date_range_text = ""

        if "Year" in df.columns and "Month" in df.columns:
            # Compute min/max YEAR
            min_year = int(df["Year"].min())
            max_year = int(df["Year"].max())

            # Filter months belonging to min year
            min_year_months = (
                df[df["Year"] == min_year]["Month"]
                .dropna()
                .unique()
                .tolist()
            )

            # Filter months belonging to max year
            max_year_months = (
                df[df["Year"] == max_year]["Month"]
                .dropna()
                .unique()
                .tolist()
            )

            # Convert month names → numbers and sort
            min_year_months_sorted = sorted(
                min_year_months,
                key=lambda m: MONTH_MAP.get(m, 13)
            )
            max_year_months_sorted = sorted(
                max_year_months,
                key=lambda m: MONTH_MAP.get(m, 13)
            )

            # Final min/max date
            if min_year_months_sorted and max_year_months_sorted:
                min_date = f"{min_year_months_sorted[0]} {min_year}"
                max_date = f"{max_year_months_sorted[-1]} {max_year}"
                date_range_text = f"Date Range from {min_date} till {max_date}"
            else:
                date_range_text = "Date range could not be determined from Month/Year values."
        else:
            date_range_text = "Date range information not available (Month or Year column missing)."

        # ---------------------------------------
        # Build SQL Generator Prompt
        # ---------------------------------------
        sql_prompt = PromptTemplate.from_template(
            """
You are an expert SQLite SQL generator.

STRICT RULES:
1. Use ONLY the columns listed in the DDL.
2. ALWAYS wrap column names in double quotes "like this".
3. NEVER use square brackets [like this] — SQLite does NOT support bracket quoting.
4. All identifiers MUST use standard SQL double quotes:
      "Product Class"
      "Month"
      "Year"
      "Sales"
5. When the user asks for:
    - last N months
    - latest year
    - latest month
    - recent month
    - recent year
you must filter the data based on the maximum Year available in the dataset.
      
6. Prefer using values that exist in the unique-values list. If a value also appears in the DATE RANGE section, you may safely use it.
7. NEVER invent months or categories.
8. Month column uses full names (January, February, ...).
9. NEVER sort months alphabetically.
10. ALWAYS use this CASE expression for month ordering and for any month-to-number conversion (including MAX(Month)):

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
Use this CASE expression whenever ordering by Month OR when calculating MAX(Month), because alphabetical comparison is invalid.

11. DO NOT assume future months or missing months.

========================
📌 DATABASE SCHEMA (DDL)
========================
{ddl}

========================
📌 SAMPLE DATA (Top 3 Rows)
========================
{sample_rows}

========================
📌 UNIQUE VALUES
========================
{dimension_values}

========================
📌 DATE RANGE
========================
{date_range_text}

========================
📌 USER QUESTION
========================
{question}

Return ONLY the SQL query. No explanation.
"""
        )

        llm = ChatOpenAI(model=model, temperature=temperature)
        generate_query = sql_prompt | llm
        execute_query = QuerySQLDataBaseTool(db=db)
        rephraser = build_rephraser(llm)
        chart_metadata_generator = build_chart_metadata_generator(llm)

        # ---------------------------------------
        # Generate SQL
        # ---------------------------------------
        sql_ai_msg = generate_query.invoke({
            "question": ensure_str(user_msg),
            "ddl": ensure_str(ddl_text),
            "sample_rows": ensure_str(sample_rows),
            "dimension_values": ensure_str(dimension_values_text),
            "date_range_text": ensure_str(date_range_text),
        })
        sql_raw = sql_ai_msg.content if hasattr(sql_ai_msg, "content") else str(sql_ai_msg)
        sql_clean = fix_month_order(clean_sql(sql_raw))

        # ---------------------------------------
        # Execute query
        # ---------------------------------------
        try:
            df_result = pd.read_sql_query(sql_clean, conn)
            result_string = df_result.to_string(index=False)
        except Exception as e:
            df_result = None
            result_string = f"SQL Error: {e}"

        # ---------------------------------------
        # Recommend Visualization Metadata (type, title, axes, colors, etc.)
        # ---------------------------------------
        chart_type = "none"
        chart_title = ""
        x_axis = y_axis = color_by = secondary_y_axis = None
        color_scheme = "plotly"
        is_time_series = False
        trendline_flag = False
        regression_type = None

        if df_result is not None and not df_result.empty:
            try:
                chart_metadata_raw = chart_metadata_generator.invoke({
                    "columns": ensure_str(list(df_result.columns)),
                    "sample": ensure_str(df_result.head(5).to_dict(orient="records")),
                })

                chart_metadata = json.loads(chart_metadata_raw)

                chart_type = str(chart_metadata.get("chart_type", "none")).lower()
                chart_title = str(chart_metadata.get("chart_title", "")).strip()

                x_axis = chart_metadata.get("x_axis")
                y_axis = chart_metadata.get("y_axis")
                color_by = chart_metadata.get("color_by")
                secondary_y_axis = chart_metadata.get("secondary_y_axis")

                color_scheme = str(chart_metadata.get("color_scheme", "plotly")).lower()
                is_time_series = bool(chart_metadata.get("is_time_series", False))
                trendline_flag = bool(chart_metadata.get("trendline", False))
                regression_type = chart_metadata.get("regression_type")

            except Exception:
                chart_type = "none"
                chart_title = ""

        # ---------------------------------------
        # Natural Language Answer
        # ---------------------------------------
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state.chat_history
        )

        answer = rephraser.invoke({
            "question": ensure_str(user_msg),
            "query": ensure_str(sql_clean),
            "result": ensure_str(result_string),
            "history": ensure_str(history_text),
        })

        # ---------------------------------------
        # Display Assistant Response + SQL + Result + Visualization
        # ---------------------------------------
        with st.chat_message("assistant"):
            # Main NL answer
            st.write(answer)

            # # SQL
            # st.write("### 🧾 Generated SQL")
            # st.code(sql_clean, language="sql")

            # SQL Result + Visualization
            st.write("### 📊 SQL Result")
            if df_result is not None and not df_result.empty:
                st.dataframe(df_result, use_container_width=True)

                # =========================================
                # ADVANCED VISUALIZATION ENGINE (Plotly)
                # =========================================
                st.write("### 📈 Visualization")

                cols = list(df_result.columns)
                num_cols = len(cols)
                cols = [str(c) for c in cols]

                numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
                cat_cols = [c for c in cols if c not in numeric_cols]

                # Pick axes safely based on LLM suggestions
                x_col = safe_col(x_axis, df_result) if x_axis else (cat_cols[0] if cat_cols else cols[0])
                y_col = safe_col(y_axis, df_result) if y_axis else (numeric_cols[0] if numeric_cols else (cols[1] if num_cols > 1 else cols[0]))
                color_col = safe_col(color_by, df_result, fallback_first=False) if color_by else (cat_cols[1] if len(cat_cols) > 1 else None)
                sec_y_col = safe_col(secondary_y_axis, df_result, fallback_first=False) if secondary_y_axis else None

                # Sorting
                df_sorted = df_result.copy()
                if is_time_series and x_col in df_sorted.columns:
                    df_sorted = df_sorted.sort_values(by=x_col, ascending=True)
                elif y_col in df_sorted.columns and y_col in numeric_cols:
                    df_sorted = df_sorted.sort_values(by=y_col, ascending=False)

                # Color schemes
                color_discrete = COLOR_QUAL.get(color_scheme)
                color_continuous = COLOR_CONT.get(color_scheme)

                fig = None
                chart_title_final = (chart_title or "Chart").title().strip()

                # -----------------------------------------
                # 2-column classic charts
                # -----------------------------------------
                if num_cols == 2 and chart_type in ["bar", "line", "area", "scatter", "pie", "histogram", "box", "violin"]:
                    a, b = cols[0], cols[1]
                    # guess which is numeric
                    if a in numeric_cols and b not in numeric_cols:
                        x_col, y_col = b, a
                    elif b in numeric_cols and a not in numeric_cols:
                        x_col, y_col = a, b
                    else:
                        x_col, y_col = cols[0], cols[1]

                    trendline_arg = regression_type if (trendline_flag and regression_type in ["ols", "lowess"] and chart_type in ["scatter", "line"]) else None

                    try:
                        if chart_type == "bar":
                            fig = px.bar(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                         color_discrete_sequence=color_discrete)

                        elif chart_type == "line":
                            fig = px.line(df_sorted, x=x_col, y=y_col, markers=True, title=chart_title_final,
                                          color_discrete_sequence=color_discrete, trendline=trendline_arg)

                        elif chart_type == "area":
                            fig = px.area(df_sorted, x=x_col, y=y_col, title=chart_title_final)

                        elif chart_type == "scatter":
                            fig = px.scatter(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                             color_discrete_sequence=color_discrete, trendline=trendline_arg)

                        elif chart_type == "pie":
                            fig = px.pie(df_sorted, names=x_col, values=y_col, title=chart_title_final,
                                         color_discrete_sequence=color_discrete)

                        elif chart_type == "histogram":
                            fig = px.histogram(df_sorted, x=y_col if y_col in numeric_cols else x_col,
                                               title=chart_title_final, color_discrete_sequence=color_discrete)

                        elif chart_type == "box":
                            fig = px.box(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                         color_discrete_sequence=color_discrete)

                        elif chart_type == "violin":
                            fig = px.violin(df_sorted, x=x_col, y=y_col, box=True, points="all",
                                            title=chart_title_final, color_discrete_sequence=color_discrete)

                    except Exception as e:
                        st.error(f"Error rendering {chart_type} chart: {e}")

                # -----------------------------------------
                # 3-column advanced charts
                # -----------------------------------------
                elif num_cols == 3 and chart_type in [
                    "grouped_bar", "stacked_bar", "treemap", "sunburst",
                    "heatmap", "bubble", "histogram", "box", "violin"
                ]:
                    dim1, dim2, metric = cols

                    # ensure metric is numeric where possible
                    if metric not in numeric_cols:
                        for c in cols:
                            if c in numeric_cols:
                                metric = c
                                break

                    try:
                        if chart_type == "grouped_bar":
                            fig = px.bar(
                                df_sorted,
                                x=dim1,
                                y=metric,
                                color=dim2,
                                barmode="group",
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )

                        elif chart_type == "stacked_bar":
                            fig = px.bar(
                                df_sorted,
                                x=dim1,
                                y=metric,
                                color=dim2,
                                barmode="relative",
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )

                        elif chart_type == "treemap":
                            fig = px.treemap(
                                df_sorted,
                                path=[dim1, dim2],
                                values=metric,
                                title=chart_title_final,
                                color=dim2,
                                color_continuous_scale=color_continuous,
                            )

                        elif chart_type == "sunburst":
                            fig = px.sunburst(
                                df_sorted,
                                path=[dim1, dim2],
                                values=metric,
                                title=chart_title_final,
                                color=dim2,
                                color_continuous_scale=color_continuous,
                            )

                        elif chart_type == "heatmap":
                            fig = px.density_heatmap(
                                df_sorted,
                                x=dim1,
                                y=dim2,
                                z=metric,
                                title=chart_title_final,
                                color_continuous_scale=color_continuous or "Blues",
                            )

                        elif chart_type == "bubble":
                            fig = px.scatter(
                                df_sorted,
                                x=dim1,
                                y=metric,
                                size=metric,
                                color=dim2,
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )

                        elif chart_type == "histogram":
                            fig = px.histogram(
                                df_sorted,
                                x=metric,
                                color=dim1,
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )

                        elif chart_type == "box":
                            fig = px.box(
                                df_sorted,
                                x=dim1,
                                y=metric,
                                color=dim2,
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )

                        elif chart_type == "violin":
                            fig = px.violin(
                                df_sorted,
                                x=dim1,
                                y=metric,
                                color=dim2,
                                box=True,
                                points="all",
                                title=chart_title_final,
                                color_discrete_sequence=color_discrete,
                            )
                    except Exception as e:
                        st.error(f"Error rendering {chart_type} chart: {e}")

                # -----------------------------------------
                # Dual-axis charts
                # -----------------------------------------
                elif chart_type in ["dual_axis_line", "dual_axis_bar"] and sec_y_col and sec_y_col in numeric_cols and y_col in numeric_cols:
                    try:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        x_vals = df_sorted[x_col]

                        # Primary
                        if chart_type == "dual_axis_bar":
                            fig.add_trace(
                                go.Bar(x=x_vals, y=df_sorted[y_col], name=y_col),
                                secondary_y=False,
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(x=x_vals, y=df_sorted[y_col], name=y_col, mode="lines+markers"),
                                secondary_y=False,
                            )

                        # Secondary
                        fig.add_trace(
                            go.Scatter(x=x_vals, y=df_sorted[sec_y_col], name=sec_y_col, mode="lines+markers"),
                            secondary_y=True,
                        )

                        fig.update_layout(title=chart_title_final)

                    except Exception as e:
                        st.error(f"Error rendering dual-axis chart: {e}")

                # -----------------------------------------
                # Fallback: scatter matrix for high-dim results
                # -----------------------------------------
                elif num_cols >= 4 and len(numeric_cols) >= 2:
                    try:
                        fig = px.scatter_matrix(df_sorted[numeric_cols], title=chart_title_final)
                    except Exception as e:
                        st.error(f"Error rendering scatter matrix: {e}")

                # -----------------------------------------
                # Final render
                # -----------------------------------------
                if fig:
                    fig.update_layout(
                        title={
                            "text": chart_title_final,
                            "x": 0.5,
                            "xanchor": "center",
                            "font": {"size": 24},
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "none":
                    st.info("The LLM indicated no meaningful chart can be created for this result.")
            else:
                st.info("No rows returned.")

        # Save assistant response text in history for future context
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Clear pending message flag
        del st.session_state["pending_user_msg"]

    # ---------------------------------------
    # Leave space so bottom messages aren't hidden behind footer
    # ---------------------------------------
    st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------
    # Load Questions from JSON for suggestions
    # ---------------------------------------
    try:
        with open("questions.json", "r") as f:
            data = json.load(f)
            QUESTIONS = data.get("common_questions", [])
    except Exception as e:
        st.error(f"Error loading questions.json: {e}")
        QUESTIONS = []

    # Generate random questions ONLY ONCE
    if "random_questions" not in st.session_state and QUESTIONS:
        st.session_state.random_questions = random.sample(QUESTIONS, min(3, len(QUESTIONS)))

    # ---------------------------------------
    # Footer: Suggested Questions + Input (fixed at bottom)
    # Only show after Excel file upload
    # ---------------------------------------
    if excel_file is not None:

        def submit_text():
            text = st.session_state.user_query.strip()
            if text:
                st.session_state.pending_user_msg = text
                st.session_state.user_query = ""
                st.rerun()

        # Sticky footer container
        st.markdown("<div class='chat-footer'><div class='chat-footer-inner'>", unsafe_allow_html=True)

        # Suggested questions block (vertical list)
        if "random_questions" in st.session_state:
            if st.session_state.random_questions:
                st.markdown("<div class='suggested-box-title'>Suggested questions</div>", unsafe_allow_html=True)

                for i, q in enumerate(st.session_state.random_questions):
                    button_key = f"suggest_btn_{i}"  # FIX: unique stable key
                    if st.button(q, key=button_key):
                        st.session_state.pending_user_msg = q
                        st.rerun()

        # Chat input box
        st.text_input(
            " ",
            placeholder="Type your message here...",
            key="user_query",
            on_change=submit_text,
            label_visibility="collapsed",
        )

        st.markdown("</div></div>", unsafe_allow_html=True)