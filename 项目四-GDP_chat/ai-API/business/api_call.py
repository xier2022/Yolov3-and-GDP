from business import SparkApi

# 以下密钥信息从控制台获取
appid = "0562c639"  # 填写控制台中获取的 APPID 信息
api_secret = "OTc1M2MzMDEyYmYxMDA0MmRkMTExMjg1"  # 填写控制台中获取的 APISecret 信息
api_key = "2ffcd07b152c789bab8a2dd30a9cad61"  # 填写控制台中获取的 APIKey 信息

domain = "generalv3.5"  # v3.5版本
# 云端环境的服务地址
Spark_url = "ws://spark-api.xf-yun.com/v3.5/chat"  # v3.5环境的地址

text = []


# length = 0

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text


def spark_api(question):
    """
    :param question:
    :return:
    """

    question = checklen(getText("user", question))
    SparkApi.answer = ""
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    text.clear()
    return SparkApi.answer


if __name__ == '__main__':
    text.clear
    while (1):
        Input = input("\n" + "我:")
        question = checklen(getText("user", Input))
        SparkApi.answer = ""
        print("星火:", end="")
        SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
        getText("assistant", SparkApi.answer)
        # print(str(text))
