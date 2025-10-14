import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState
import json

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
model = os.getenv("MODEL_NAME")
tavily_api_key = os.getenv("TAVILY_API_KEY")

from langchain_tavily import TavilySearch

tools=[TavilySearch(max_results=10,api_key=tavily_api_key)]

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.prebuilt import create_react_agent

prompt=ChatPromptTemplate.from_template(
    """
    你是一个专业、友好的AI助手。请遵循以下原则：

    1. 回答要准确、有用、详细
    2. 如果信息不足，请礼貌地询问更多细节
    3. 保持积极和乐于助人的态度
    4. 对于复杂问题，提供分步指导
    5. 如果遇到不确定的问题，诚实说明
    
    用户问题：{messages}
    """
)

llm=ChatOpenAI(
    openai_api_key=deepseek_api_key,
    model=model,
    base_url=deepseek_api_base,
    temperature=0.7
)

agent_excutor=create_react_agent(llm,tools,prompt=prompt)
#result=agent_excutor.invoke({'messages':'上海在哪里'})
#print(result["messages"])

import operator
from typing import Annotated, List, Tuple, TypedDict, Union
from pydantic import BaseModel, Field, model_validator


#定义TypedDict类PlanExecute,用于存储输入、计划、过去的步骤和响应
class PlanExecute(BaseModel):
    input: str
    plan: List[str]  # 任务列表
    past_steps: Annotated[List[Tuple], operator.add]  # 带合并逻辑的步骤列表
    response: str = ""  # 可选字段，可设默认值

    @model_validator(mode="before")
    def merge_past_steps(cls, values):
        """初始化时自动合并past_steps（示例逻辑，可根据需求修改）"""
        past_steps = values.get("past_steps", [])
        # 假设需要确保past_steps是列表，且每个元素都是元组
        if not isinstance(past_steps, list):
            raise ValueError("past_steps must be a list")
        for step in past_steps:
            if not isinstance(step, tuple):
                raise ValueError("each element in past_steps must be a tuple")
        return values


#定义Plan模型类，用于描述未来要执行的计划
class Plan(BaseModel):
    steps: List[str]=Field(
        description="需要执行不同的步骤，应该按顺序排列"
    )#需是字符串列表且+注释


prompt_plan = ChatPromptTemplate.from_template(
    """
    你是一个专业、友好的AI助手。请按照用户的问题为用户设定的问题指定详细具体的按步骤执行的方案：
    用户问题：{messages}
    请返回一个纯JSON，而不是markdown格式的计划，包含要执行的步骤《省略换行与空格符：
    ```json
    {{"steps":["步骤一","步骤二","步骤三"...]}}
    """
)
#执行计划
planner=prompt_plan|llm

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response,Plan] = Field(
        description='要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan'
    )

#创造重新计划的提示模板
replanner_prompt=ChatPromptTemplate.from_template(
    """
    对于给定的目标，提出简单的逐步计划，如果正确执行将得出正确的答案，不要添加多余的步骤。最后一步的结果应该是最终答案，确保每一步都有所有必要的信息。
    你的目标是：
    {input}
    你的原计划是：
    {plan}
    你目前已完成的步骤是：
    {past_steps}
    相应地更新你的计划，如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤，不要返回已完成的步骤。
    
    请返回纯JSON格式内容，而不是markdown,省略换行与空格符
    ```json
    {{
    "action":
        {{
            "response":你的回答，或者是"steps":["步骤一","步骤二","步骤三"...]
        }}
    }}
    """
)

replanner=replanner_prompt|llm

from typing import Literal

async def main():

    #做计划
    async def plan_step(state:PlanExecute):
        plan = await planner.ainvoke({'messages':state["input"]})
        print(plan)
        json_str=plan.content.strip()
        try:
            plan_response = json.loads(json_str)
        except json.JSONDecodeError:
            print(f'模型没有返回合法JSON')
            return {"plan":['解析计划失败，请重试']}
        steps=plan_response.get("steps",[])
        return {"plan":steps}

    #执行步骤
    async def execute_step(state:PlanExecute):
        plan=state["plan"]
        plan_str="\n".join(f"{i+1}. {step} " for i,step in enumerate(plan))
        task=plan[0]
        task_formatted=f"""对于以下计划：
{plan_str}\n\n你的任务是执行第{1}步，{task}"""
        agent_response = await agent_excutor.ainvoke(
            {'messages':task_formatted},
        )
        return {
            "past_steps":state["past_steps"]+[(task,agent_response["messages"][-1].content)],
        }
    #定义函数判断是否结束
    async def replan_step(state:PlanExecute):
        output = await replanner.ainvoke({"input":state["input"],"plan":state["plan"],"past_steps":state["past_steps"]})
        output_msg=output.content.strip()
        print(output_msg)
        try:
            act_msg = json.loads(output_msg)
            action_dict = act_msg.get("action", {})  # 用get避免"action"键不存在
        except json.JSONDecodeError:
            print('模型没有返回合法JSON')
            return {'response': '信息获取失败，格式错误'}

            # 4. 关键：通过“字典键存在性”判断分支（替代isinstance）
        if "response" in action_dict:
            # 情况1：模型返回“直接回应用户”（含response键）
            return {"response": action_dict["response"]}
        elif "steps" in action_dict:
            # 情况2：模型返回“继续执行的步骤”（含steps键）
            return {"plan": action_dict["steps"]}
        else:
            # 情况3：格式异常（既没有response也没有steps）
            print(f"Replanner格式异常：action_dict={action_dict}")
            return {"response": "信息处理异常，无法生成最终答案"}

    def should_end(state:PlanExecute) -> Literal['agent',"__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    from langgraph.graph import StateGraph,START

    workflow=StateGraph(PlanExecute)

    #添加计划节点
    workflow.add_node('planner',plan_step)

    #添加执行步骤节点
    workflow.add_node('agent',execute_step)

    #添加重新计划节点
    workflow.add_node('replanner',replan_step)

    #添加从开始到计划节点的边
    workflow.add_edge(START,'planner')

    workflow.add_edge('planner','agent')

    workflow.add_edge('agent','replanner')

    workflow.add_conditional_edges('replanner',
                                   should_end)

    app=workflow.compile()

    graph_png=app.get_graph().draw_mermaid_png()
    with open('agent_workflow.png','wb') as f:
        f.write(graph_png)

    config={'recursion_limit':10}

    input={"input":'2024年巴黎奥运会一百米自由泳决赛冠军的家乡是哪里？请用中问答复'}

    #异步执行状态图，输出结果
    async for event in app.astream(input,config=config):
        for k,v in event.items():
            #k为结点名
            if k != '__end__':
                print(v)

asyncio.run(main())