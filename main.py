import streamlit as st
import base64
from pathlib import Path
from appRCA import app
# -----------------------------------------
# Encode Image to Base64
# -----------------------------------------
def get_base64_image(image_path):
    if not Path(image_path).exists():
        st.error(f"Image not found: {image_path}")
        return None

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        mime = "image/png" if image_path.lower().endswith("png") else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None


# -----------------------------------------
# Authentication
# -----------------------------------------
def authenticate(username, password):
    return username == "Bizinsight" and password == "Bizinsight@2024"


# -----------------------------------------
# Login Page
# -----------------------------------------
def login_page():
    bg = get_base64_image("logo.png")

    if bg:
        st.markdown(
            f"""
            <style>
            /* Background Container */
            [data-testid="stAppViewContainer"] {{
                background: url("{bg}") no-repeat center center fixed;
                background-size: cover;
            }}

            /* Add slight dark overlay for readability */
            [data-testid="stAppViewContainer"]::before {{
                content: "";
                position: absolute;
                top: ; left: 0;
                width: 100%; height: 100%;
                background: rgba(0,0,0,0.35); /* Adjust transparency */
                z-index: 0;
            }}

            /* Push content above overlay */
            .block-container {{
                position: relative;
                z-index: 5;
            }}

            /* Style input boxes */
            input[type="text"], input[type="password"] {{
                background-color: #ffffffd9;
                color: black;
                border-radius: 6px;
                padding: 20px;
                font-size: 16px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h1 style='text-align: center; color: white;'>Login</h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")


# -----------------------------------------
# Main App
# -----------------------------------------
def main():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
        return
    app()


# -----------------------------------------
# Entry Point
# -----------------------------------------
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    main()
