# app.py — Excel → SQLite Version (Conversational + Metadata-Aware SQL Engine)

def app():
    import os
    import re
    import json
    import sqlite3
    import pandas as pd
    import streamlit as st
    from dotenv import load_dotenv

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from langchain_community.utilities import SQLDatabase
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

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
    # Chat History
    # ---------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

        m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)

        s = re.sub(r"^\s*(sql\s*query|sqlquery)\s*:\s*", "", s, flags=re.IGNORECASE)

        lines = s.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("select", "with")):
                s = "\n".join(lines[i:])
                break

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
            st.rerun()

    # ---------------------------------------
    # Render Chat History
    # ---------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---------------------------------------
    # Chat Input
    # ---------------------------------------
    user_msg = st.chat_input("Ask something about your Excel data...")
    if not user_msg:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    if excel_file is None:
        with st.chat_message("assistant"):
            st.error("Upload an Excel file first.")
        return

    # ---------------------------------------
    # Load Excel → SQLite
    # ---------------------------------------
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name if sheet_name else 0)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error reading Excel: {e}")
        return

    db_path = "excel_chat.db"
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # ---------------------------------------
    # Metadata Extraction
    # ---------------------------------------
    def generate_ddl(df):
        cols = []
        for col, dtype in df.dtypes.items():
            if "int" in str(dtype):
                sql_type = "INTEGER"
            elif "float" in str(dtype):
                sql_type = "FLOAT"
            else:
                sql_type = "TEXT"
            cols.append(f'"{col}" {sql_type}')
        return "CREATE TABLE excel_data (\n  " + ",\n  ".join(cols) + "\n);"

    ddl_text = generate_ddl(df)
    sample_rows = ensure_str(df.head(5).to_dict(orient="records"))

    dimension_values = {}
    for col in df.columns:
        if df[col].dtype == object:
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

    # Available months (raw list)
    available_months = (
        df["Month"].dropna().unique().tolist()
        if "Month" in df.columns else []
    )

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

        # Convert month names → numbers
        min_year_months_sorted = sorted(min_year_months, key=lambda m: MONTH_MAP[m])
        max_year_months_sorted = sorted(max_year_months, key=lambda m: MONTH_MAP[m])

        # Final min/max date
        min_date = f"{min_year_months_sorted[0]} {min_year}"
        max_date = f"{max_year_months_sorted[-1]} {max_year}"

        # Build final text
        date_range_text = f"Min Date: {min_date}\nMax Date: {max_date}"

    else:
        date_range_text = "Min Date: N/A\nMax Date: N/A"
        

    # ---------------------------------------
    # Build SQL Generator Prompt
    # ---------------------------------------
    sql_prompt = PromptTemplate.from_template(
        """
You are an expert SQLite SQL generator.

STRICT RULES:
1. Use ONLY the columns listed in the DDL.
2. Use ONLY values that exist in the unique-values list.
3. NEVER invent months or categories.
4. Month column uses full names (January, February, ...).
5. NEVER sort months alphabetically.
6. ALWAYS use this CASE expression for month ordering:

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

7. When asked about:
   - last N months
   - recent months
   - latest quarter
   Compute these ONLY using the available date available in the dataset:
   Min Date: January 2022
   Max Date: November 2025

8. DO NOT assume future months or missing months.

========================
📌 DATABASE SCHEMA (DDL)
========================
{ddl}

========================
📌 SAMPLE DATA (Top 5 Rows)
========================
{sample_rows}

========================
📌 UNIQUE VALUES
========================
{dimension_values}

========================
📌 DATE RANGE
========================
Min Date: January 2022
Max Date: November 2025

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

    # ---------------------------------------
    # Generate SQL
    # ---------------------------------------
    sql_ai_msg = generate_query.invoke({
        "question": ensure_str(user_msg),
        "ddl": ensure_str(ddl_text),
        "sample_rows": ensure_str(sample_rows),
        "dimension_values": ensure_str(dimension_values_text)
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
    # Natural Language Answer
    # ---------------------------------------
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[:-1]
    )

    answer = rephraser.invoke({
        "question": ensure_str(user_msg),
        "query": ensure_str(sql_clean),
        "result": ensure_str(result_string),
        "history": ensure_str(history_text),
    })

    # ---------------------------------------
    # Display Assistant Response
    # ---------------------------------------
    with st.chat_message("assistant"):
        st.write(answer)

        st.write("### 🧾 Generated SQL")
        st.code(sql_clean, language="sql")

        st.write("### 📊 SQL Result")
        if df_result is not None and not df_result.empty:
            st.dataframe(df_result, use_container_width=True)
        else:
            st.info("No rows returned.")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
