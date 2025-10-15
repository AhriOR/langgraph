from operator import add
from typing import TypedDict,Annotated
from langgraph.constants import END
from langgraph.graph import StateGraph,MessagesState,START

class State(TypedDict):
    messages:Annotated[list[str],add]

#subgraph
def sub_node_1(state:State)->MessagesState:
    return {'messages':['response from subgraph']}

subgraph_builder=StateGraph(State)
subgraph_builder.add_node('sub_node_1',sub_node_1)
subgraph_builder.add_edge(START,"sub_node_1")
subgraph_builder.add_edge('sub_node_1',END)

subgraph=subgraph_builder.compile()

#parent graph
builder=StateGraph(State)
builder.add_node('subgraph_node',subgraph)
builder.add_edge(START,"subgraph_node")
builder.add_edge('subgraph_node',END)
graph=builder.compile()

print(graph.invoke({'messages':['hello subgraph']}))

for chunk in graph.stream({'messages':['hello subgraph']},stream_mode='debug'):
    print(chunk)