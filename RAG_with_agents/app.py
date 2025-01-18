import streamlit as st
from login_signup import LogInSignUp
from dashboard_with_agents import Dashboard


# Check if user is logged in
if "username" not in st.session_state:
    LogInSignUp().show()
else:
    Dashboard().show()