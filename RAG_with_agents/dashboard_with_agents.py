import streamlit as st
from pymongo import MongoClient
from rag_with_agents import RAGApplicationWithAgents
from models import OPENAI_LLM, GEMINI_LLM
from dotenv import load_dotenv
import os



class Dashboard:
    def __init__(self):
        load_dotenv()
        # MongoDB connection setup
        self.client = MongoClient(os.environ.get("MONGO_CONNECTION_STRING"))
        self.db = self.client["financial-chat-bot"]
        self.history_collection = self.db["history"]
        user_history = self.history_collection.find_one({"user_id": st.session_state["username"]})
        print(user_history)
        if user_history:
            st.session_state["session_id"] = len(user_history["history"])-1
        else:
            st.session_state["session_id"] = 0


    def initialize_history(self, user_id):
        user_record = self.history_collection.find_one({"user_id": user_id})
        if not user_record:
            # Create user entry with the first session 
            self.history_collection.insert_one({
                "user_id": user_id,
                "history": [{"session_id": 1, "conversation": []}]
            })
        elif not user_record.get("history"):
            # Add the first session if history is empty
            self.history_collection.update_one(
                {"user_id": user_id},
                {"$set": {"history": [{"session_id": 1, "conversation": []}]}}
            )


    def append_conversation(self, user_id, session_id, user_query, ai_response):
        # Find the user's history
        user_history = self.history_collection.find_one({"user_id": user_id})
        if not user_history:
            raise ValueError("User not found")

        # Find the index of the session with the given session_id
        session_index = None
        for index, session in enumerate(user_history["history"]):
            if session["session_id"] == session_id:
                session_index = index
                break

        if session_index is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")

        # Update the conversation for the specific session
        self.history_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    f"history.{session_index}.conversation": {
                        "$each": [f"you: {user_query}", f"ai: {ai_response}"]
                    }
                }
            }
        )



    def fetch_conversation_history(self, user_id, session_id):
        user_history = self.history_collection.find_one({"user_id": user_id})
        if user_history:
            for session in user_history["history"]:
                if session["session_id"] == session_id:
                    return session["conversation"]
        return []

    
    def fetch_session_list(self, user_id):
        user_history = self.history_collection.find_one({"user_id": user_id})
        sessions_list = []
        if user_history:
            for session in user_history["history"]:
                if session["conversation"]:
                    sessions_list.append(session["conversation"][0])
                else:
                    sessions_list.append("New Chat")
        return sessions_list


    def create_new_session(self, user_id):
        user_history = self.history_collection.find_one({"user_id": user_id})
        new_session_id = len(user_history["history"]) + 1 if user_history else 1
        new_session = {"session_id": new_session_id, "conversation": []}
        self.history_collection.update_one(
            {"user_id": user_id},
            {"$push": {"history": new_session}},
            upsert=True
        )
        return new_session_id


    def show(self):
        st.subheader(f"Welcome, {st.session_state['username']}!")
        user_id = st.session_state["username"]
        session_id = st.session_state["session_id"]
        self.initialize_history(user_id)

        with st.sidebar:

            st.header("Chat Application")

            st.write("-"*50)
            model = st.selectbox("Select Model", ["gpt-4o", "Gemini-1.5"])

            st.write("-"*50)
            st.subheader("Chat Sessions")
            if st.button("Add New Chat"):
                new_session_id = self.create_new_session(user_id)
                st.success(f"New session created: Session {new_session_id}")
                st.rerun()

            # Display Sessions
            sessions_list = self.fetch_session_list(user_id)

            # Radio button to select an session
            selected_item = st.radio("Select a session to view conversation history. ", sessions_list)

            # Get the index of the selected session
            if selected_item:
                session_id = sessions_list.index(selected_item)
                st.session_state["session_id"] = session_id


            # Logout button
            if st.button("Logout"):
                st.session_state.pop("username", None)
                st.session_state.pop("session_id", None)
                st.rerun()


        # Display conversation history
        conversation_history = self.fetch_conversation_history(user_id, session_id + 1)
        for message in conversation_history:
            st.write(message)

        query = st.text_input("Enter text")
        if st.button("Submit") and query.strip():

            rag = RAGApplicationWithAgents()
            print("RAG Application Initialized")
            results, aggregated_results, rewritten_query = rag.run_context_aware_query(query, conversation_history)
            relavent_documents = []
            for index, response in results.items():
               documents = [node.text for node in response.source_nodes]
               for doc in documents:
                    relavent_documents.append(doc)
                
            st.write(conversation_history)

            """"Integrating LLMs Model"""
            if model == "gpt-4o":
                ai_response = OPENAI_LLM().get_response(conversation_history, relavent_documents, rewritten_query)
                st.write("Openai")
            elif model == "Gemini-1.5":
                ai_response = GEMINI_LLM().get_response(conversation_history, relavent_documents, rewritten_query)
                st.write("Gemini")
                
            st.write(f"Rewritten_query: {rewritten_query}")
            st.write(f"User: {query}")
            st.write(f"AI: {ai_response}")

            self.append_conversation(user_id, session_id+1, query, ai_response)
            st.success("Conversation saved to history!")
            st.success("Text saved to history!")
            

            
        