import streamlit as st
from pymongo import MongoClient
import hashlib
import os
from dotenv import load_dotenv



class LogInSignUp:
    def __init__(self):
        load_dotenv()
        # MongoDB connection setup
        self.client = MongoClient(os.environ.get("MONGO_CONNECTION_STRING"))
        self.db = self.client["financial-chat-bot"]
        self.auth_collection = self.db["auth"]
        self.history_collection = self.db["history"]
        print(f"Connected to MongoDB database: {self.db.name} and collections: {self.auth_collection.name}, {self.history_collection.name}")


    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    

    def verify_login(self, username, password):
        hashed_password = self.hash_password(password)
        user = self.auth_collection.find_one({"username": username, "password": hashed_password})
        return user is not None
    

    def create_user(self, username, password):
        hashed_password = self.hash_password(password)
        if self.auth_collection.find_one({"username": username}):
            return False
        self.auth_collection.insert_one({"username": username, "password": hashed_password})
        return True
    

    def show(self):
        st.title("Signup and Login")
        # Tabs for Signup and Login
        tab1, tab2 = st.tabs(["Signup", "Login"])
        with tab1:
            st.header("Signup")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Signup"):
                if self.create_user(new_username, new_password):
                    st.success("User created successfully!")
                else:
                    st.error("Username already exists")
        with tab2:
            st.header("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if self.verify_login(username, password):
                    st.success("Login successful!")
                    st.session_state["username"] = username
                else:
                    st.error("Invalid username or password")