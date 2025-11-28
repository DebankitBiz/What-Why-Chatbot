def app():
    import os
    import re
    import json
    import sqlite3
    import random
    import warnings
    import time

    import pandas as pd
    import streamlit as st
    from dotenv import load_dotenv

    # Load environment variables BEFORE LangChain loads
    load_dotenv()

    # Force LangSmith on
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "StructuredDataAssistant")

    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from langchain_community.utilities import SQLDatabase
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

    from visual import visual, plot_combined_rca
    from prompt import sql_prompt, viz_prompt, answer_prompt, classify_question_type
    from rca import rca_agent, run_rca

    warnings.filterwarnings("ignore")

    st.set_page_config(page_title="Structured Data Assistant", page_icon="📊", layout="wide")

    st.markdown(
        "<h1 style='text-align: center;'>📊 Structured Data Assistant Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 16px;'>Upload an Excel file and chat with your data in natural language.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* Hide Streamlit system warning area */
        .stAlert { display: none !important; }
        .stNotification { display: none !important; }
        .st-emotion-cache-1wqrz03 { display: none !important; }  /* generic Streamlit warning banner */
        </style>
        """,
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

        .stTextInput > div > div {
            margin-bottom: 0px !important;
        }

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
        s = re.sub(
            r"^\s*(sql\s*query|sqlquery)\s*:\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )

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
        return PromptTemplate.from_template(answer_prompt) | llm | StrOutputParser()

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

    # ---------------------------------------
    # Combined Chart Metadata Generator
    # ---------------------------------------
    def build_chart_metadata_generator(llm: ChatOpenAI):
        return PromptTemplate.from_template(viz_prompt) | llm | StrOutputParser()

    # ---------------------------------------
    # Sidebar Settings
    # ---------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        excel_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
        #sheet_name = st.text_input("Sheet Name (optional)", "")
        sheet_name=""
        table_name = "excel_data"

        #model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0)
        model="gpt-4o-mini"
        #temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        temperature=0.0

        # api_key = st.text_input("OPENAI_API_KEY", type="password")
        # if api_key:
        #     os.environ["OPENAI_API_KEY"] = api_key

        st.divider()
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            if "pending_user_msg" in st.session_state:
                del st.session_state["pending_user_msg"]
            if "why_buffer_chart" in st.session_state:
                del st.session_state["why_buffer_chart"]
            st.rerun()

    # ---------------------------------------
    # Render previous Chat History (with charts)
    # ---------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg.get("content", ""))

            if msg["role"] == "assistant":
                # Restore table if present
                df_saved = msg.get("df_result")
                if df_saved is not None:
                    try:
                        df_prev = pd.DataFrame(df_saved)
                        st.dataframe(df_prev, use_container_width=True)
                    except Exception:
                        pass

                # Restore chart if present (WHAT flow only; WHY does not save chart)
                chart_json = msg.get("chart")
                if chart_json:
                    try:
                        st.write("### 📈 Visualization")
                        fig_prev = pio.from_json(chart_json)
                        st.plotly_chart(fig_prev, use_container_width=True)
                    except Exception:
                        pass
        msg["df_result"] = None
        msg["chart"] = None

    # ---------------------------------------
    # Render buffered WHY chart from last run (if any)
    # ---------------------------------------
    if "why_buffer_chart" in st.session_state:
        try:
            with st.chat_message("assistant"):
                fig_buf = pio.from_json(st.session_state["why_buffer_chart"])
                st.plotly_chart(fig_buf, use_container_width=True)
        except Exception:
            pass
        # Clear the buffer so it only shows once
        del st.session_state["why_buffer_chart"]

    # ---------------------------------------
    # If there is a new message to process (typed or clicked)
    # ---------------------------------------
    if pending_msg is not None:
        start_all = time.time()
        user_msg = pending_msg

        # ✅ Only append user to history, don't render directly
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        # If no Excel file, answer and rerun (keep footer visible)
        if excel_file is None:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Please upload an Excel file first."}
            )
            del st.session_state["pending_user_msg"]
            st.rerun()

        # ---------------------------------------
        # Cache: Load Excel → SQLite ONCE per uploaded file
        # ---------------------------------------
        # We store: cached_df, db_conn, db, cached_file_id
        if "cached_file_id" not in st.session_state or st.session_state.cached_file_id != getattr(excel_file, "name", None):
            t0 = time.time()
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name if sheet_name else 0)
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error reading Excel: {e}"}
                )
                del st.session_state["pending_user_msg"]
                st.rerun()

            db_path = "excel_chat.db"
            # Use a persistent connection and allow multithread access for Streamlit
            conn = sqlite3.connect(db_path, check_same_thread=False)
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            # Create lightweight indices on first few columns (if sensible)
            try:
                cols_for_idx = [c for c in df.columns[:5]]
                for col in cols_for_idx:
                    try:
                        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}("{col}");')
                    except Exception:
                        pass
            except Exception:
                pass

            # Create SQLDatabase wrapper once
            try:
                db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            except Exception:
                db = None

            st.session_state.cached_df = df
            st.session_state.db_conn = conn
            st.session_state.db = db
            st.session_state.cached_file_id = getattr(excel_file, "name", None)

            # Precompute metadata used in prompts (keep small)
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

            st.session_state.ddl_text = generate_ddl(df)
            st.session_state.sample_rows = ensure_str(df.head(3).to_dict(orient="records"))

            obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
            dim_vals = {col: df[col].dropna().unique().tolist()[:10] for col in obj_cols}
            st.session_state.dimension_values_text = ensure_str(dim_vals)

            # Date range text
            MONTH_MAP = {
                "January": 1,
                "February": 2,
                "March": 3,
                "April": 4,
                "May": 5,
                "June": 6,
                "July": 7,
                "August": 8,
                "September": 9,
                "October": 10,
                "November": 11,
                "December": 12,
            }
            df_local = st.session_state.cached_df
            if "Year" in df_local.columns and "Month" in df_local.columns:
                try:
                    min_year = int(df_local["Year"].min())
                    max_year = int(df_local["Year"].max())
                    min_year_months = df_local[df_local["Year"] == min_year]["Month"].dropna().unique().tolist()
                    max_year_months = df_local[df_local["Year"] == max_year]["Month"].dropna().unique().tolist()
                    min_year_months_sorted = sorted(min_year_months, key=lambda m: MONTH_MAP.get(m, 13))
                    max_year_months_sorted = sorted(max_year_months, key=lambda m: MONTH_MAP.get(m, 13))
                    if min_year_months_sorted and max_year_months_sorted:
                        min_date = f"{min_year_months_sorted[0]} {min_year}"
                        max_date = f"{max_year_months_sorted[-1]} {max_year}"
                        st.session_state.date_range_text = f"Date Range from {min_date} till {max_date}"
                    else:
                        st.session_state.date_range_text = "Date range could not be determined from Month/Year values."
                except Exception:
                    st.session_state.date_range_text = "Date range could not be determined from Month/Year values."
            else:
                st.session_state.date_range_text = "Date range information not available (Month or Year column missing)."

            t1 = time.time()
            # small log
            # st.experimental_set_query_params(_cache_load_time=round(t1-t0,2))

        # Reuse cached artifacts
        df = st.session_state.cached_df
        conn = st.session_state.db_conn
        db = st.session_state.db

        # ---------------------------------------
        # Reuse LLM instance (store in session_state)
        # ---------------------------------------
        if "llm" not in st.session_state:
            st.session_state.llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            # If model selection changed, recreate llm
            existing_llm = st.session_state.llm
            try:
                if getattr(existing_llm, "model_name", None) != model:
                    st.session_state.llm = ChatOpenAI(model=model, temperature=temperature)
            except Exception:
                pass

        llm = st.session_state.llm

        # Python function classifier (not LLM)
        question_type = classify_question_type(user_msg).lower()

        # ----------------------------------------------------
        # IF QUESTION TYPE = WHY → RUN RCA PIPELINE (cached)
        # ----------------------------------------------------
        if "why" in question_type:
            # simple caching for WHY queries
            why_cache = st.session_state.setdefault("why_cache", {})
            q_key = user_msg.strip().lower()
            if q_key in why_cache:
                agent_output, combined_fig_json = why_cache[q_key]
            else:
                try:
                    rca_text, failure_type = run_rca(user_msg,df)
                    agent_output = rca_agent(user_msg, rca_text, failure_type)
                    combined_fig = plot_combined_rca(rca_text)
                    combined_fig_json = combined_fig.to_json() if combined_fig is not None else None
                except Exception as e:
                    agent_output = f"RCA Error: {e}"
                    combined_fig_json = None
                why_cache[q_key] = (agent_output, combined_fig_json)

            # Save ONLY text in history (no chart)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": agent_output,
                    "df_result": None,
                    "chart": None,
                    "type": "why",
                }
            )

            # Store chart in temporary buffer to render after rerun
            if combined_fig_json is not None:
                st.session_state["why_buffer_chart"] = combined_fig_json

            # Clear pending message and rerun to render from history + buffer
            del st.session_state["pending_user_msg"]
            st.rerun()

        # ---------------------------------------
        # Metadata Extraction (for WHAT flow)
        # ---------------------------------------
        ddl_text = st.session_state.ddl_text
        sample_rows = st.session_state.sample_rows
        dimension_values_text = st.session_state.dimension_values_text
        date_range_text = st.session_state.date_range_text

        # ---------------------------------------
        # Build SQL Generator Prompt (WHAT flow)
        # ---------------------------------------
        # Reuse PromptTemplate objects (cheap) but avoid re-instantiating LLM
        sql_prompt_template = PromptTemplate.from_template(sql_prompt)
        generate_query = sql_prompt_template | llm
        execute_query = QuerySQLDataBaseTool(db=db) if db is not None else None
        rephraser = build_rephraser(llm)
        chart_metadata_generator = build_chart_metadata_generator(llm)

        # ---------------------------------------
        # Generate SQL (single LLM call)
        # ---------------------------------------
        # We keep prompt inputs small to reduce tokens
        sql_inputs = {
            "question": ensure_str(user_msg),
            "ddl": ensure_str(ddl_text),
            "sample_rows": ensure_str(sample_rows),
            "dimension_values": ensure_str(dimension_values_text),
            "date_range_text": ensure_str(date_range_text),
        }

        sql_ai_msg = generate_query.invoke(sql_inputs)
        sql_raw = sql_ai_msg.content if hasattr(sql_ai_msg, "content") else str(sql_ai_msg)
        sql_clean = fix_month_order(clean_sql(sql_raw))

        # ---------------------------------------
        # Execute query (with memoization)
        # ---------------------------------------
        query_cache = st.session_state.setdefault("query_cache", {})
        cache_key = sql_clean
        if cache_key in query_cache:
            df_result = query_cache[cache_key]
        else:
            try:
                # limit result size for interactivity
                df_result = pd.read_sql_query(sql_clean, conn)
                if len(df_result) > 5000:
                    # keep a head for interactive display and cache smaller copy
                    query_cache[cache_key] = df_result.head(1000)
                else:
                    query_cache[cache_key] = df_result
            except Exception as e:
                df_result = None
                result_string = f"SQL Error: {e}"

        if df_result is not None:
            result_string = df_result.to_string(index=False)

        # ---------------------------------------
        # Recommend Visualization Metadata (reuse LLM but small payload)
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
                chart_metadata_raw = chart_metadata_generator.invoke(
                    {
                        "columns": ensure_str(list(df_result.columns)),
                        "sample": ensure_str(df_result.head(5).to_dict(orient="records")),
                    }
                )
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
        # Natural Language Answer (WHAT flow)
        # ---------------------------------------
        history_text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in st.session_state.chat_history)

        answer = rephraser.invoke(
            {
                "question": ensure_str(user_msg),
                "query": ensure_str(sql_clean),
                "result": ensure_str(result_string),
                "history": ensure_str(history_text),
            }
        )

        # ---------------------------------------
        # Build Visualization (WHAT flow)
        # ---------------------------------------
        fig = None
        if df_result is not None and not df_result.empty:
            fig = visual(
                df_result,
                x_axis,
                y_axis,
                color_by,
                secondary_y_axis,
                chart_type,
                chart_title,
                color_scheme,
                trendline_flag,
                regression_type,
                COLOR_QUAL,
                COLOR_CONT,
                make_subplots,
            )

        # ---------------------------------------
        # Save assistant response + table + chart in history (WHAT)
        # ---------------------------------------
        msg_assistant = {
            "role": "assistant",
            "content": answer,
            "df_result": (df_result.to_dict(orient="list") if df_result is not None else None),
            "chart": fig.to_json() if fig is not None else None,
            "type": "what",
        }
        st.session_state.chat_history.append(msg_assistant)

        # Clear pending message and rerun to render results from history
        del st.session_state["pending_user_msg"]
        # small timing log (optional)
        # st.experimental_set_query_params(_last_request_sec=round(time.time()-start_all,2))
        st.rerun()

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
    except Exception:
        QUESTIONS = []

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

        st.markdown("<div class='chat-footer'><div class='chat-footer-inner'>", unsafe_allow_html=True)

        if "random_questions" in st.session_state:
            if st.session_state.random_questions:
                st.markdown("<div class='suggested-box-title'>Suggested questions</div>", unsafe_allow_html=True)

                for i, q in enumerate(st.session_state.random_questions):
                    button_key = f"suggest_btn_{i}"
                    if st.button(q, key=button_key):
                        st.session_state.pending_user_msg = q
                        st.rerun()

        st.text_input(
            " ",
            placeholder="Type your message here...",
            key="user_query",
            on_change=submit_text,
            label_visibility="collapsed",
        )

        st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    app()

