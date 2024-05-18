# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import json
""" 
  人脸特征分析颜值WebAPI接口调用示例接口文档(必看)：https://doc.xfyun.cn/rest_api/%E4%BA%BA%E8%84%B8%E7%89%B9%E5%BE%81%E5%88%86%E6%9E%90-%E9%A2%9C%E5%80%BC.html
  图片属性：png、jpg、jpeg、bmp、tif图片大小不超过800k
  (Very Important)创建完webapi应用添加服务之后一定要设置ip白名单，找到控制台--我的应用--设置ip白名单，如何设置参考：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=41891
  错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
  @author iflytek
"""
# 人脸特征分析颜值webapi接口地址
URL = "http://tupapi.xfyun.cn/v1/face_score"
# 应用ID  (必须为webapi类型应用，并人脸特征分析服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
APPID = "0562c639"
# 接口密钥 (webapi类型应用开通人脸特征分析服务后，控制台--我的应用---人脸特征分析---服务的apikey)
API_KEY = "b1561c1f7ef648103c492e705c55a02d"
ImageName = "1.png"

#ImageUrl = "http://hbimg.b0.upaiyun.com/a09289289df694cd6157f997ffa017cc44d4ca9e288fb-OehMYA_fw658"
FilePath = r"D:\flask\flaskProject2\images\1.png"
# 图片数据可以通过两种方式上传，第一种在请求头设置image_url参数，第二种将图片二进制数据写入请求体中。若同时设置，以第一种为准。
# 此demo使用第一种方式进行上传图片地址，如果想使用第二种方式，将图片二进制数据写入请求体即可。
def getHeader(image_name, image_url=None):
    curTime = str(int(time.time()))
    param = "{\"image_name\":\"" + image_name + "\"}"
    paramBase64 = base64.b64encode(param.encode('utf-8'))
    tmp = str(paramBase64, 'utf-8')

    m2 = hashlib.md5()
    m2.update((API_KEY + curTime + tmp).encode('utf-8'))
    checkSum = m2.hexdigest()

    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
    }
    return header



def getBody(filePath):
    binfile = open(filePath, 'rb')
    data = binfile.read()
    return data




def run():
    # 发送请求
    r = requests.post(URL, headers=getHeader(ImageName), data=getBody(FilePath))
    print(r.content)
    # 将响应内容解析为JSON格式
    result_json = json.loads(r.content)

    # 返回JSON格式的结果到前端
    return json.dumps(result_json)



if __name__ == '__main__':
    run()


