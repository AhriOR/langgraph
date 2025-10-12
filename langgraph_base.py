from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END,StateGraph,MessagesState
from langgraph.prebuilt import ToolNode

@tool
def search(query:str):
    "查询天气"
    return '天气很好'

tools=[search]

tool_node=ToolNode(tools)

model=ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-395fccc24f79470b871ca11723da78d6",
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.8,
).bind_tools(tools)

def should_continue(state:MessagesState) -> Literal["tools",END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return 'tools'
    return END


def call_model(state:MessagesState):
    messages = state["messages"]
    response= model.bind_tools(tools).invoke(messages)
    return {'messages':[response]}

workflow=StateGraph(MessagesState)

workflow.add_node('agent',call_model)
workflow.add_node('tools',tool_node)

workflow.set_entry_point('agent')

workflow.add_conditional_edges(
    'agent',
    should_continue,
    {
        'tools': 'tools',
        END: END
    }
)

workflow.add_edge('tools','agent')

checkpointer=MemorySaver()#redis

app=workflow.compile(checkpointer=checkpointer)

final_state=app.invoke({'messages':[HumanMessage(content='上海天气怎么样')]},
                       config={'configurable':{'thread_id':42}}
                       )
result=final_state['messages'][-1].content
print(result)
final_state=app.invoke({'messages':[HumanMessage(content='我问的是哪个城市')]},
                       config={'configurable':{'thread_id':42}})
result=final_state['messages'][-1].content
print(result)