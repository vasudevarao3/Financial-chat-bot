import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

class OPENAI_LLM:
    def __init__(self):
        load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        print(OPENAI_API_KEY)
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

        self.llm = OpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)

                

    def get_response(self, history, prompt):

        qa_prompt = f"""
            You are an AI assisstent that answers prompt.

            ### Conversation History:
            {history}

            ### User Prompt:
            {prompt}

            ### Guidelines for Answer:
            - If the prompt is not clear, rewrite the prompt based on the conversational history and answer the prompt.
        """
        print("*"*50,"QA Prompt: ", "*"*50,"\n", qa_prompt)

        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content= "You are helpyou AI assisstent"
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







