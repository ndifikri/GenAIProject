import streamlit as st

def login_page():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username == st.secrets.user[username].username and password == st.secrets.user[username].password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success("Logged in successfully!")
                # You can redirect to another page or display content here
            else:
                st.error("Invalid username or password")

def main_app():
    st.title("Welcome to the Main App!")
    st.write(f"Hello, {st.session_state.get('username', 'Guest')}!")
    if st.button("Logout"):
        del st.session_state["logged_in"]
        del st.session_state["username"]
        st.experimental_rerun() # Rerun to show login page

if "logged_in" not in st.session_state:
    login_page()
else:
    main_app()