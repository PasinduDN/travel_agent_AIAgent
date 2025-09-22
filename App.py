from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import load_tools
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from operator import add as add_messages
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

os.environ["OPENAI_API_KEY"] = "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"

#Initialize Open Whether map tool
os.environ["OPENWEATHERMAP_API_KEY"] = "47df7aa72aeb68ff373a395e593655ef"

chat_sessions = {}

app = FastAPI()  # Your existing FastAPI app

# Add this immediately after creating `app`
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

#Define AgentState
class AgentState(TypedDict):
    # This annotation tells LangGraph to append new messages to the existing list
    messages: Annotated[Sequence[BaseMessage], add_messages]
    preferences_text: str  # Stores the full collected text from frontend

# Define the request body
class PreferencesRequest(BaseModel):
    preferences_text: str
    location: str
    start_date: str  # Use YYYY-MM-DD format
    end_date: str

@tool
def destination_tool(preferences: str) -> str:
    """Suggest 3-5 destinations in Sri Lanka based on user preferences."""
    system_prompt = """
    You are a travel assistant for Sri Lanka.
    Based on the user's preferences, suggest 3-5 destinations.
    Respond only with a list of destinations separated by commas.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def accommodation_tool(preferences: str) -> str:
    """
    Generate recommended accommodations in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a travel assistant for Sri Lanka.
    Based on the user's accommodation preferences, suggest hotels, resorts, lodges, or guesthouses.
    Include 3-5 options.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def food_tool(preferences: str) -> str:
    """
    Suggest food options in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a Sri Lanka travel assistant.
    Suggest 3-5 restaurants, food types, or dishes that match the user's food preferences.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def activity_tool(preferences: str) -> str:
    """
    Suggest activities in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a Sri Lanka travel assistant.
    Based on the user's activity preferences, suggest 3-5 relevant activities or tours.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def weather_tool(location: str, travel_date: str, preferences: str = "") -> str:
    """
    Suggest locations in Sri Lanka based on weather preferences.
    """
    weather_api = OpenWeatherMapAPIWrapper()
    forecast = weather_api.get_weather(location, travel_date)
    
    # Use LLM to filter locations based on forecast + user preferences
    system_prompt = f"""
    You are a travel assistant for Sri Lanka.
    The forecast for {location} on {travel_date} is: {forecast}.
    Based on the user's weather preferences ({preferences}), suggest 1-3 locations suitable for visiting.
    """
    messages = [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return response.content

#LLM Agent Function
def call_llm(state: AgentState) -> AgentState:
    """
    Calls LLM to process the user's preferences text
    and decide which tools to call.
    """
    # The system prompt can be added at the beginning of the chat session
    # For simplicity here, we ensure it's present before calling the LLM
    messages = state['messages']
    
    system_prompt = """
    You are an intelligent AI travel assistant for tourists in Sri Lanka.
    You have access to the following tools:
    - destination_tool
    - accommodation_tool
    - food_tool
    - activity_tool
    - weather_tool
    Based on the full conversation history, decide if a tool is needed or if you can answer the user.
    If you have tool results, synthesize them into a final answer.
    """
    
    # Check if system prompt is already there to avoid adding it multiple times
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Call LLM
    message = llm.invoke(messages)
    
    # The 'add_messages' reducer will append this new message to the state
    return {'messages': [message]}


#Tool Execution Function 
def take_action(state: AgentState) -> AgentState:
    tools_dict = {
        "destination_tool": destination_tool,
        "accommodation_tool": accommodation_tool,
        "food_tool": food_tool,
        "activity_tool": activity_tool,
        "weather_tool": weather_tool
    }
    
    results = []
    tool_calls = getattr(state['messages'][-1], 'tool_calls', [])
    
    for t in tool_calls:
        tool_name = t['name']
        args = t['args']
        if tool_name in tools_dict:
            result = tools_dict[tool_name].invoke(**args)
        else:
            result = f"Tool {tool_name} not found."
        results.append(ToolMessage(tool_call_id=t.get('id', ""), name=tool_name, content=str(result)))
    
    return {'messages': results}

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool_agent", take_action)

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", should_continue, {True: "tool_agent", False: END})
workflow.add_edge("tool_agent", "llm")

compiled_workflow = workflow.compile()

@app.post("/api/process_preferences/{session_id}")
async def process_preferences(session_id: str, request: PreferencesRequest):
    if not request.preferences_text.strip() or not request.location.strip() or not request.start_date.strip() or not request.end_date.strip():
        raise HTTPException(status_code=400, detail="Please provide all fields.")
    
    full_preferences = f"Location: {request.location}. Travel Dates: {request.start_date} to {request.end_date}. Preferences: {request.preferences_text}"
    
    # Retrieve previous messages for this session
    previous_messages = chat_sessions.get(session_id, [])
    previous_messages.append(HumanMessage(content=full_preferences))
    
    state = {
        "messages": previous_messages,
        "preferences_text": full_preferences
    }
    
    result = compiled_workflow.invoke(state)
    
    # Save new messages to session
    previous_messages.append(result['messages'][-1])
    chat_sessions[session_id] = previous_messages
    
    return {"result": result['messages'][-1].content}


# # Then at the bottom, for testing:
# test_preferences = "Beach, Eco-Lodge, Local Sri Lankan, Hiking & Nature Trails"

# result = compiled_workflow.invoke({
#     "messages": [HumanMessage(content="Process preferences")],
#     "preferences_text": test_preferences
# })

# print(result)  # see the full output structure
