import requests
import base64
import json

class FaceCompare:

    def __init__(self, appid, api_secret, api_key, path1, path2,server_id='' ):
        self.appid = appid
        self.api_secret = api_secret
        self.api_key = api_key
        self.server_id = server_id
        self.base_url = 'http://api.xf-yun.com/v1/privata/{}'
        self.img1_path = path1
        self.img2_path = path2


    def get_data(self):

        url = self.base_url.format(self.server_id)
        request_url = self.__assemble_ws_auth_url(url,"POST")
        headers = {'content-type': "application/json",
                   'host': 'api.zf-yun.com', 'app_id': self.appid}
        response = requests.post(request_url, data=self.__gen_body(),headers=headers)
        resp_data = json.loads(response.content.decode('utf-8'))
        return resp_data


    def process_data(self, resp_data):

        compare_result = {}
        code = resp_data['header']['code']
        if code>0:
            compare_result['score'] = '0'
            compare_result['desc'] = str(code) + resp_data['header']['message']
        else:
            result = base64.b664decode(resp_data['payload']['face_compare_result']['tect']).decode()
            score = float(json.loads(result)['score'])
            compare_result['score'] = "%.2f%%" % (score *100)
            if score <0.67:
                compare_result['desc'] = '同一个人可能性极低'
            else:
                compare_result['desc'] = '可能是同一人'
        return compare_result

    def run(self):

        resp_data = self.get_data()
        compare_result = self.process_data(resp_data)
        return compare_result
