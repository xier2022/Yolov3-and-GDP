import datetime
import json
import logging

logging.basicConfig(level=logging.WARNING)

import os
from threading import Event
from sparkai.socket_mode.websocket_client import SparkAISocketModeClient
from sparkai.memory import ChatMessageHistory

print_question = False

response_format = {
    "thoughts": {
        "text": "thought",
        "speak": "thoughts summary to say to user",
        "plan": "- short bulleted - list that conveys - long-term plan",
        "reasoning": "reasoning"
    }
}
rf = json.dumps(response_format, indent=4)

question = ""
query_prompt = f'''
帮我润色下如下问题:

{question}

'''

from sparkai.prompts.classification import PROMPTS

query_prompt1 = f'''
总结下述问题并按照如下json格式输出:
{rf}

请注意回答的结果必须满足下述约束:
1. 结果响应只能包含json内容
2. 结果响应不能有markdown内容
3. 结果中json格式务必正确且能够被python json.loads 解析

现在请回答: {question}

'''
if __name__ == "__main__":
    client = SparkAISocketModeClient(
        app_id=os.environ.get("APP_ID"),
        api_key=os.environ.get("API_KEY"),
        api_secret=os.environ.get("API_SECRET"),
        chat_interactive=False,
        trace_enabled=False,
        conversation_memory=ChatMessageHistory()
    )

    q = PROMPTS + "帮我发送一份邮件给 ybyang7@iflytek.com, 内容由你帮我生成一段写进去，主要表达欢迎他加入公司的意思就可以"
    # q = PROMPTS + "2023年5月8日，合肥天气怎么样"
    if print_question:
        print("Question: ", q)
    client.connect()
    result = client.chat_with_histories(
        [
            {'role': 'user', 'content': '请帮我完成目标:\n\n帮我生成一个 2到2000的随机数\n\n'}, {'role': 'assistant',
                                                                               'content': '{\n\n"thoughts": {\n\n"text": "Generate a random number between 2 and 2000.",\n\n"reasoning": "To complete this task, I will need to access the internet for information gathering.",\n\n"plan": "I will use the random_number command with the min and max arguments set to 2 and 2000, respectively.",\n\n"criticism": "",\n\n"speak": "The random number generated is: 1587."\n\n},\n\n"command": {\n\n"name": "random_number",\n\n"args": {\n\n"min": "2",\n\n"max": "2000"\n\n}\n\n}\n\n}'},
            {'role': 'user', 'content': '\n请帮我完成目标:\n\n帮我把这个随机数 发给 ybyang7@iflytek.com 并告诉他这个随机数很重要\n\n'}])

    if result:
        print(result.content)
    #  result = client.chat_in("你是谁")

    Event().wait()
