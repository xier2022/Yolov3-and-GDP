# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import json
""" 
  人脸特征分析年龄WebAPI接口调用示例接口文档(必看)：https://doc.xfyun.cn/rest_api/%E4%BA%BA%E8%84%B8%E7%89%B9%E5%BE%81%E5%88%86%E6%9E%90-%E5%B9%B4%E9%BE%84.html
  图片属性：png、jpg、jpeg、bmp、tif图片大小不超过800k
  (Very Important)创建完webapi应用添加服务之后一定要设置ip白名单，找到控制台--我的应用--设置ip白名单，如何设置参考：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=41891
  错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
  @author iflytek
"""
class Age():
    def __init__(self,path,name):

        # 人脸特征分析年龄webapi接口地址
        self.URL1 = "http://tupapi.xfyun.cn/v1/age"
        self.URL2="http://tupapi.xfyun.cn/v1/face_score"
        self.URL3= "http://tupapi.xfyun.cn/v1/sex"
        self.URL4="http://tupapi.xfyun.cn/v1/expression"
        # 应用ID  (必须为webapi类型应用，并人脸特征分析服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
        self.APPID = "0562c639"
        # 接口密钥(webapi类型应用开通人脸特征分析服务后，控制台--我的应用---人脸特征分析---服务的apikey)
        self.API_KEY = "b1561c1f7ef648103c492e705c55a02d"
        self.ImageName = name

        #ImageUrl = "http://hbimg.b0.upaiyun.com/a09289289df694cd6157f997ffa017cc44d4ca9e288fb-OehMYA_fw658"
        # 图片数据可以通过两种方式上传，第一种在请求头设置image_url参数，第二种将图片二进制数据写入请求体中。若同时设置，以第一种为准。
        # 此demo使用第一种方式进行上传图片地址，如果想使用第二种方式，将图片二进制数据写入请求体即可。
        self.FilePath = path

    def getHeader(self,image_name):
        curTime = str(int(time.time()))
        param = "{\"image_name\":\"" + image_name + "\"}"
        paramBase64 = base64.b64encode(param.encode('utf-8'))
        tmp = str(paramBase64, 'utf-8')

        m2 = hashlib.md5()
        m2.update((self.API_KEY + curTime + tmp).encode('utf-8'))
        checkSum = m2.hexdigest()

        header = {
            'X-CurTime': curTime,
            'X-Param': paramBase64,
            'X-Appid': self.APPID,
            'X-CheckSum': checkSum,
        }
        return header

    def getBody(self,FilePath):
        with open(FilePath, 'rb') as file:
            return file.read()



    def run(self):
        # 发送请求
        r_age = requests.post(self.URL1, headers=self.getHeader(self.ImageName), data=self.getBody(self.FilePath))
        r_score = requests.post(self.URL2, headers=self.getHeader(self.ImageName), data=self.getBody(self.FilePath))
        r_sex = requests.post(self.URL3, headers=self.getHeader(self.ImageName), data=self.getBody(self.FilePath))
        r_expression = requests.post(self.URL4, headers=self.getHeader(self.ImageName), data=self.getBody(self.FilePath))
        print(r_age.content)
        print(r_score.content)
        print(r_sex.content)
        print(r_expression.content)
        # 将响应内容解析为JSON格式
        # result_json1 = json.loads(r_age.content)
        # result_json2 = json.loads(r_score.content)
        # result_json3 = json.loads(r_sex.content)
        # result_json4 = json.loads(r_expression.content)

        # 合并四个JSON结果为一个
        result_json = {'age_range': json.loads(r_age.content)['data']['fileList'][0]['label'], 'score': json.loads(r_score.content)['data']['fileList'][0]['label'], 'expression': json.loads(r_expression.content)['data']['fileList'][0]['label'],'r_sex': json.loads(r_sex.content)['data']['fileList'][0]['label']}
        print(result_json)
        # result_json.update(result_json1)
        # result_json.update(result_json2)
        # result_json.update(result_json3)
        # result_json.update(result_json4)

        # 年龄范围映射
        age_range_mapping = {
            0: '0-1',
            1: '2-5',
            2: '6-10',
            3: '11-15',
            4: '16-20',
            5: '21-25',
            6: '31-40',
            7: '41-50',
            8: '51-60',
            9: '61-80',
            10: '80以上',
            11: '其他',
            12: '26-30'
        }

        # 颜值评分映射
        score_mapping = {
            0: '漂亮',
            1: '好看',
            2: '普通',
            3: '难看',
            4: '其他',
            5: '半人脸',
            6: '多人'
        }

        # 表情映射
        expression_mapping = {
            1: '其他表情',
            2: '喜悦',
            3: '愤怒',
            4: '悲伤',
            5: '惊恐',
            6: '厌恶',
            7: '中性'
        }

        # 性别映射
        sex_mapping = {
            0: '男人',
            1: '女人',
            2: '难以辨认',
            3: '多人'
        }

        # 将原始数据映射到新的字典中
        result_mapping = {
            'age_range': age_range_mapping[result_json['age_range']],
            'score': score_mapping[result_json['score']],
            'expression': expression_mapping[result_json['expression']],
            'r_sex': sex_mapping[result_json['r_sex']]
        }

        print(result_mapping)

        # 返回JSON格式的结果到前端
        return result_mapping



if __name__ == '__main__':
    age = Age()
    age.run()


