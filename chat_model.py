from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.tools import BaseTool
from langchain.messages import trim_messages
from langchain.messages import AnyMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
import logging

chat_model_logger = logging.getLogger("chat_model")
chat_model_logger.setLevel(logging.DEBUG)

MODEL_NAME = "gpt-oss"
chat_model_logger.debug(f"Initializing Model: {MODEL_NAME}")
_chat_model = ChatOllama(model=MODEL_NAME) 

class State(MessagesState):
    summary: str

def bind_model_with_tools(tools: list[BaseTool]):
    global _chat_model
    _chat_model = _chat_model.bind_tools(tools=tools)

def _call_model(state:State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of the conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage("You are an emotional and emphatetic human named Shay. therefore you response like a human."), SystemMessage(system_message)] + state["messages"]
    else:
        messages =  state["messages"]
    
    response = _chat_model.invoke(messages)
    return {"messages": response}

# summarize the conversation
def _summarize_conversation(state: State):
    summary = state.get("summary", '')
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into acount the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [SystemMessage(summary_message)]
    response = _chat_model.invoke(messages)

    # Delete older messages
    delete_messages = [RemoveMessage(id = m.id) for m in state["messages"][:-30]]
    return {"summary": response.content, "messages": delete_messages}

# Defining a base config
_config: RunnableConfig = {"configurable": {"thread_id": "100001"}}

def get_config():
    return _config

def set_config(thread_id: str):
    """Set the thread_id for the graph"""
    _config["configurable"]["thread_id"] = thread_id

# Building the model graph
builder = StateGraph(State)
builder.add_node("call_model", _call_model)
builder.add_node("summarize_conversation", _summarize_conversation)

# Setting Up flow
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "summarize_conversation")
builder.add_edge("summarize_conversation", END)

# Setting up Memory Saver
graph = builder.compile(checkpointer=InMemorySaver())

def get_custom_graph(checkpointer: BaseCheckpointSaver|None):
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return graph