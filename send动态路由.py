from operator import add
from typing import  TypedDict,Annotated
from langgraph.constants import START,END
from langgraph.graph import StateGraph
from langgraph.types import Send



class State(TypedDict):
    message : Annotated[list[str],add]

class PrivateState(TypedDict):
    msg:str

def node(state:PrivateState) -> State:
    res=state['msg']+"!"
    return {"message":[res]}

builder=StateGraph(State)

builder.add_node('node_1',node)

def routing_func(state:State):
    result=[]
    for msg in state['message']:
        result.append(Send('node_1',{'msg':msg}))
    return result
builder.add_conditional_edges(START,routing_func,"node_1")

builder.add_edge('node_1',END)

graph=builder.compile()

print(graph.invoke({"message":['hello,"world','hello','graph']}))


#将state中更新状态再通过edges传递给下一个node的两个步骤合并为同一个命令
from langgraph.types import Command

class State(TypedDict):
    message : Annotated[list[str],add]

class PrivateState(TypedDict):
    msg:str

def node1(state:State):
    new_message=[]
    for msg in state['message']:
        new_message.append(msg+"!")
    return Command(
        goto=END,
        update={'message':new_message}
    )

builder=StateGraph(State)

builder.add_node('node_2',node1)


builder.add_edge(START,"node_2")

graph=builder.compile()

print(graph.invoke({"message":['hello,"world','hello','graph']}))