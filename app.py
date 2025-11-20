# app.py — Excel → SQLite Version (Conversational + DataFrame Output)
"""
Project: Excel NL→SQL Runner (Streamlit)
Version: 2.0.0
Maintainer: Debankit
Python: 3.10+
"""

def app():
    import os
    import re
    import sqlite3
    import pandas as pd
    import streamlit as st
    from dotenv import load_dotenv

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

    # ---------------------------------------
    # Load ENV, Page Setup
    # ---------------------------------------
    load_dotenv()
    st.set_page_config(page_title="Excel NL→SQL Agent", page_icon="📊", layout="wide")

    st.markdown(
        "<h1 style='text-align: center;'>📊 Structured Data Assistant Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 16px;'>Upload an Excel file and chat with your data using natural language. I will generate SQL, execute it, and answer clearly.</p>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------
    # Init Chat History
    # ---------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # stores {"role": "user"/"assistant", "content": ""}

    # ---------------------------------------
    # Helper: Clean SQL
    # ---------------------------------------
    def clean_sql(s: str) -> str:
        """Extract runnable SQL from LLM output."""
        if not s:
            return ""

        # extract content inside ```sql ... ```
        fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        if fence_match:
            s = fence_match.group(1)

        # remove prefixes
        s = re.sub(r"^\s*(sql\s*query|sqlquery)\s*:\s*", "", s, flags=re.IGNORECASE)

        # strip text before actual SELECT/WITH
        lines = s.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("select", "with")):
                s = "\n".join(lines[i:])
                break

        # remove stray fences + semicolons
        s = s.replace("```", "").strip().rstrip(";")
        return s
    
    def fix_month_order(sql):
    # mapping for month name sorting
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

        # replace ORDER BY "Month"
        sql = re.sub(
            r'ORDER BY\s+"Month"\s*(ASC|DESC)?',
            rf'ORDER BY {month_case} \1',
            sql,
            flags=re.IGNORECASE
        )

        return sql
    # ---------------------------------------
    # Helper: Rephraser Chain
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

IMPORTANT MONTH RULES:
- The "Month" column contains FULL month names (e.g., "January", "February", ..., "December").
- The values are NOT numeric (not 01 or 02).
- Whenever SQL needs to sort or compare months, ALWAYS convert month names to numbers using:

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

- NEVER order months alphabetically.
- NEVER compare month names without converting them.

IMPORTANT:
- Do NOT use LaTeX, formulas, \text, \frac, or math notation.
- Always write clean, readable explanations.
- Show calculations in simple English, like:
    November Sales: X
    October Sales: Y
    Growth = ((X - Y) / Y) * 100 = Z%
- If it is a growth question, provide a bullet summary.

Provide the final answer in a friendly, readable format relevant to the user's question."""
        )
        | llm
        | StrOutputParser()
    )

    # ---------------------------------------
    # Sidebar Config
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
    # Show chat conversation so far
    # ---------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---------------------------------------
    # Chat Input
    # ---------------------------------------
    user_msg = st.chat_input("Ask something about your Excel data...")

    if user_msg:
        # store user message
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        # show user bubble
        with st.chat_message("user"):
            st.write(user_msg)

        # Validate file
        if excel_file is None:
            with st.chat_message("assistant"):
                st.error("Please upload an Excel file first.")
            return

        # ---------------------------------------
        # Read Excel + Load to SQLite
        # ---------------------------------------
        df = pd.read_excel(excel_file, sheet_name=sheet_name if sheet_name else 0)
        db_path = "excel_chat.db"
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        # ---------------------------------------
        # Initialize LLM + SQL Components
        # ---------------------------------------
        llm = ChatOpenAI(model=model, temperature=temperature)
        generate_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDataBaseTool(db=db)
        rephraser = build_rephraser(llm)

        # Prepare conversation history string
        history_txt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[:-1]])

        # ---------------------------------------
        # Generate SQL
        # ---------------------------------------
        sql_raw = generate_query.invoke({
            "question": user_msg,
            "top_k": 1000,
            "table_names_to_use": [table_name],
        })
        sql_clean = clean_sql(sql_raw)
        sql_clean = fix_month_order(sql_clean)

        # ---------------------------------------
        # Execute SQL
        # ---------------------------------------
        result_string = execute_query.invoke(sql_clean)

        # Try converting to DataFrame
        try:
            df_result = pd.read_sql_query(sql_clean, conn)
        except Exception:
            df_result = None

        # ---------------------------------------
        # Generate NL Answer
        # ---------------------------------------
        answer = rephraser.invoke({
            "question": user_msg,
            "query": sql_clean,
            "result": result_string,
            "history": history_txt
        })

        # ---------------------------------------
        # Display assistant response
        # ---------------------------------------
        with st.chat_message("assistant"):
            st.write(answer)

            st.write("### 🧾 Generated SQL")
            st.code(sql_clean, language="sql")

            st.write("### 📊 Query Result (Table)")
            if df_result is not None and not df_result.empty:
                st.dataframe(df_result, use_container_width=True)
            else:
                st.info("Query executed successfully, but no rows returned.")

        # append assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
