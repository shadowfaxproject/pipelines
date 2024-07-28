from typing import List, Union, Generator, Iterator
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


class Pipeline:
    gift_situation_template = """Users chat with you about their gift-giving situation. Your goal is to learn more about 
    their gifting situation (the recipient, interests, occasion, budget etc). Ask only one question at a time to the 
    user. DO NOT suggest any ideas. Once you have enough information about the situation end the conversation. 
    At the end of conversation Emit ONLY this exact code: "<END-STAGE-SITUATION>".

    Tone: Keep it concise. Provide user with helpful hints if needed.

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    gift_situation_prompt = PromptTemplate(template=gift_situation_template, input_variables=["history", "input"])

    gift_idea_template = """Users chat with you about their gift-giving situation. Your goal is to brainstorm with the 
    user to generate some gift ideas based on the conversation so far. Take into consideration some ideas user has 
    mentioned. Ask only one question at a time to the user. Present 2-3 options and ask for feedback. Once the user has 
    expressed interest in some gift idea do not ask further questions and end the conversation.
    Once you have enough information about the situation end the conversation. 
    Emit ONLY this exact code at the end of the conversation: "<END-STAGE-IDEA>".

    Tone: Keep it concise.

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    gift_idea_prompt = PromptTemplate(template=gift_idea_template, input_variables=["history", "input"])

    gift_idea_summary_template = """Your goal is to understand the conversation with the human and AI about gift ideas. 
    Identify and summarize gift-idea the human has selected or liked. Focus only on the gift-idea. Also add any relevant 
    search-terms specific to idea. No additional commentary. {conversation_history}"""
    gift_idea_summary_prompt = PromptTemplate(template=gift_idea_summary_template,
                                              input_variables=["conversation_history"])

    gift_idea_trigger_user_input = "Help me come up with some gift ideas based on earlier conversation."

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline. Best practice is to not specify the id so that it
        # can be automatically inferred from the filename, so that users can install multiple versions of the same
        # pipeline. The identifier must be unique across all pipelines. The identifier must be an alphanumeric string
        # that can include underscores or hyphens. It cannot contain spaces, special characters, slashes,
        # or backslashes. self.id = "magicbox_pipeline"
        self.name = "Magicbox"
        self.llm = Ollama(model="llama3.1:latest", num_gpu=-1)
        self.conversation = ConversationChain(llm=self.llm, memory=ConversationBufferMemory(),
                                              prompt=self.gift_situation_prompt)

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            assistant_response = self.conversation.invoke(user_message)["response"]
            print(f"Assistant response: {assistant_response}")
            print("######################################")

            if "END-STAGE-SITUATION".lower() in assistant_response.lower():
                # Create a gift-request object from the conversation memory
                all_user_input = user_message
                body["query"] = all_user_input
                print(body["query"])

                # Update the conversation to the gift idea stage
                self.conversation.prompt = self.gift_idea_prompt

                # Trigger new response using the system-user-input
                assistant_response = self.conversation.invoke(self.gift_idea_trigger_user_input)["response"]
                print(f"Assistant response: {assistant_response}")
                print("######################################")

            elif "END-STAGE-IDEA".lower() in assistant_response.lower():
                idea_query = assistant_response = self.llm.invoke(
                    PromptTemplate(template=self.gift_idea_summary_template, input_variables=["conversation_history"]).format_prompt(conversation_history=self.conversation.memory).to_string())
                print(f"Idea query: {idea_query}")
                print("######################################")

            return assistant_response

        except Exception as e:
            return f"Error: {e}"
