import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
import google.generativeai as genai


class OPENAI_LLM:
    def __init__(self):
        load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

        self.llm = OpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)

                

    def get_response(self, history, relevant_documents, query):
        context = ""
        for doc in relevant_documents:
            context += doc + "\n"

        qa_prompt = f"""
            You are an AI assisstent[you are aware of Infosys, Uber and Alteryx companies financial reports] that answers questions strictly based on the provided Relevants Documents and conversation history, if available.
            If no context is available or no context is used to answer, respond with 'No relevant information found.'

            ### Conversation History:
            {history}

            ### Relevant Documents:
            {context}

            ### User Query:
            {query}

            ### Guidelines for Answer:
            1. Provide the **Final Answer** concisely without displaying the step-by-step reasoning process or irrelevant details.
            2. If the user query does not specify a year, provide data for all available years mentioned in the context.
            3. If the user query does not specify a industry name, provide data for all available industries mentioned in the context 
            4. Include an **Explanation** of how the relevant documents helped in answering the query, focusing on why the context is relevant.
            5. If no relevant documents are found, explicitly state: 'No relevant information found.'
            6. If query context is out of the box, just say sorry and I am not aware about it.
        """
        print("*"*50,"QA Prompt: ", "*"*50,"\n", qa_prompt)

        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content= "You are helpful AI assisstent"
                ),
            ChatMessage(role=MessageRole.USER, content= qa_prompt),
        ]

         # Creating Chat Prompt Template
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
        # print("Creating ChatPromptTemplate:", text_qa_template)

        #QA Response from formatted msgs
        response = text_qa_template.format_messages(chat_text_qa_msgs)
        # print("*"*50,"QA Response: ", "*"*50,"\n", qa_response)

        print(response)

        ai_response = self.llm.chat(response)

        return ai_response.message.blocks[0].text.strip()
    

class GEMINI_LLM:
    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key is missing. Ensure it is set in the .env file.")

        genai.configure(api_key=GEMINI_API_KEY)
        self.llm = genai.GenerativeModel(model_name = "gemini-1.5-flash")
        print("Gemini is Created")   


    def get_response(self, history, relevant_documents, query):
        context = ""
        for doc in relevant_documents:
            context += doc + "\n"

        qa_prompt = f"""
            You are an AI assisstent[you are aware of Infosys, Uber and Alteryx companies financial reports] that answers questions strictly based on the provided Relevants Documents and conversation history, if available.
            If no context is available or no context is used to answer, respond with 'No relevant information found.'

            ### Conversation History:
            {history}

            ### Relevant Documents:
            {context}

            ### User Query:
            {query}

            ### Guidelines for Answer:
            1. Provide the **Final Answer** concisely without displaying the step-by-step reasoning process or irrelevant details.
            2. If the user query does not specify a year, provide data for all available years mentioned in the context.
            3. If the user query does not specify a industry name, provide data for all available industries mentioned in the context 
            4. Include an **Explanation** of how the relevant documents helped in answering the query, focusing on why the context is relevant.
            5. If no relevant documents are found, explicitly state: 'No relevant information found.'
            6. If query context is out of the box, just say sorry and I am not aware about it.
        """
        print("*"*50,"QA Prompt: ", "*"*50,"\n", qa_prompt)

        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content= "You are helpful AI assisstent"
                ),
            ChatMessage(role=MessageRole.USER, content= qa_prompt),
        ]

         # Creating Chat Prompt Template
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
        # print("Creating ChatPromptTemplate:", text_qa_template)

        #QA Response from formatted msgs
        response = text_qa_template.format_messages(chat_text_qa_msgs)
        # print("*"*50,"QA Response: ", "*"*50,"\n", qa_response)

        print(response)

        ai_response = self.llm.generate_content(response[1].blocks[0].text)
        return ai_response.text.strip()







