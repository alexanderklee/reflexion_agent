from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END
from langgraph.graph.message import MessageGraph
from langgraph import Me

from chains import revisor, first_responder
from tool_executor import execute_tools

from dotenv import load_dotenv

load_dotenv()

# Limit x iterations to critique and iterations
MAX_ITERATIONS=2

builder = MessageGraph()
builder.add_node(key:"draft", first_responder)
builder.add_note(key:"execute_tools", execute_tools)
builder.add_node(key:"revise", revisor)

builder.add_edge(start_key:"draft", end_key:"execute_tools")
builder.add_edge(start_key:"execute_tools", end_key:"revise")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS: 
        return END
    return "execute_tools"

builder.add_conditional_edges(start_key:"revise", event_loop)
builder.set_entry_point("draft")

    
if __name__ == '__main__':
    print("hello world")