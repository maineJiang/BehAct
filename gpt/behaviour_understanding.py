import openai
import os
from IPython.display import display, Markdown, Latex
from langchain.llms import OpenAI
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

openai.api_key = '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'

# system prompt，用于告诉GPT当前的情景，不了解可以放空，没有影响。
# system prompt例如：'You are a marketing consultant, please answer the client's questions in profession style.'
# system_content = ""
with open('scene_prompt.txt', 'r') as f:
    system_content = f.read()



# 这里使用了langchain包简化与GPT的对话过程，基于的是GPT-3.5，能力与免费版的chatGPT相同。GPT-4需要自行申请加入waitlist
messages = [SystemMessage(content=system_content)]

# 一轮最多对话20次，防止过长的对话。可以通过while循环条件修改。
i = 1
while i <= 20:
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)

    user_input = input('type to input')

    # 输入\end结束
    if user_input == '\end':
        break
    # 输入\clear清空当前对话重来，重置对话场景
    if user_input == '\clear':
        i = 1
        messages = [SystemMessage(content=system_content)]
        continue

    messages.append(HumanMessage(content=user_input))

    response = chat(messages)

    messages.append(AIMessage(content=response.content))  # 将GPT回复加入到对话

    print("[GPT] Round " + str(i))
    print(response.content)
    display(Markdown(response.content))

    i = i + 1

print("\n --- END ---")