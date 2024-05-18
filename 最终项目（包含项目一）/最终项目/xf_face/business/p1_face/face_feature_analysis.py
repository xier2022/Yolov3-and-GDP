import json
import requests
class FaceFeature:
    def __init__(self,APPID,API_KEY,path):
        """
        构造函数，初始化数据
        :param APPID: 应用ID
        :param API_KEY: 接口密钥
        :param path: 图片路径
        """
        self.APPID = APPID
        self.API_key = API_KEY
        self.base_url = "http://tupai.xfyun.cn/v1/"   #基础URL
        self.image_path = path
        self.mode = 0     #图片路径模式：0本地图片，1网络图片


def  __get_data_by_type(self,type,headers,data=None):


    try:
        result = requests.post(self.base_url + type,data=data, headers=headers)
        result = json.loads(result.content)
        code = result['code']
        if code == 0:
            label = result['data']['filelist'][0]['label']
        else:
            label = result['desc']
    except:
        code = -1
        label = '校验类型%s是否正确' % type
    return code,label


def get_data(self):

    res = []
    data = self.__get_body()
    headers = self.__get_header()
    code,age = self.__get_data_by_type('age', headers,data)
    res.append({'type':'age','code':code,'value':age})
    code,face_score = self.__get_data_by_type('face_score', headers, data)
    res.append({'type': 'face_score', 'code': code, 'value':face_score})
    code, sex = self.__get_data_by_type('sex',headers,data)
    res.append({'type': 'sex','code': code, 'value': sex})
    code, exp = self.__get_data_by_type('expression',headers,data)
    res.append({'type': 'exp', 'code':code,'value':exp})
    return res


class FaceDesc:
    def __init__(self, value):
        self.value = value

    def convert_age(self):
        # 在这里实现年龄转换的逻辑
        pass

def process_data(self, res):
    process_result = []
    for item in res:
        if item['type']==('age'):
            if item['code'] == 0:
                item['value'] = FaceDesc(item['value']).convert_age()
            process_result.append({'type':'性别', 'desc':item['value']})
        elif item['type'] == ('face_score'):
            if item['code'] == 0:
                item['value'] = FaceDesc(item['value']).convert_age()
            process_result.append({'type':'颜值', 'desc':item['value']})
        elif item['type'] == ('sex'):
            if item['code'] == 0:
                item['value'] = FaceDesc(item['value']).convert_age()
            process_result.append({'type': '性别', 'desc': item['value']})
        else:
            if item['code'] == 0:
                item['value'] = FaceDesc(item['value']).convert_age()
            process_result.append({'type': '表情', 'desc': item['value']})
        return process_result

        return code,label



def face_local_analysis(self):

    self.mode = 0
    request_data = self.get_data()
    process_data = self.process_data(request_data)
    return process_data
