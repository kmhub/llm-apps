from langchain_openai import ChatOpenAI
import streamlit as st
import os
os.environ["OPENAI_API_KEY"] = "  "  # provide your API key from https://platform.openai.com/api-keys

from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)

from langchain.callbacks import StreamlitCallbackHandler
# This function returns AgentExecutor which is a chain
def load_agent() -> AgentExecutor:

    # streaming = True will result in better user experience since it means that the text response will be updated as it comes in, rather than once all the text has been completed
    llm = ChatOpenAI(model="gpt-4o",temperature=0, streaming=True)

    # DuckDuckGo: A search engine that focuses on privacy; an added advantage is that it doesn’t require developer signup
    # Wolfram Alpha: An integration that combines natural language understanding with math capabilities, for questions like “What is 2x+5 = -3x + 7?”
    # arXiv: Search in academic pre-print publications; this is useful for research-oriented questions
    # Wikipedia: For any question about entities of significant notoriety

    tools = load_tools(
        tool_names=["ddg-search", "arxiv", "wikipedia"], # "wolfram-alpha" - requires a key
        llm=llm
    )

    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )

chain = load_agent()

st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)
