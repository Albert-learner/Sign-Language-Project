from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import urllib.parse as urlparse
import re
import torch
import torch.nn.functional as F
from model import LeNet5, ResNet
import numpy as np

host = '203.246.113.83' # 호스트 ip를 적어주세요
port = 8080             # 포트번호를 임의로 설정해주세요

call_back_counter = 0
valid_call_back_counter = 0

data_list = []

idx_to_label = {
    0 : '-',
    1 : 'ㄱ',
    2 : 'ㄴ',
    3 : 'ㄷ',
    4 : 'ㄹ',
    5 : 'ㅁ',
    6 : 'ㅂ',
    7 : 'ㅅ',
    8 : 'ㅇ',
    9 : 'ㅈ',
    10 : 'ㅊ',
    11 : 'ㅋ',
    12 : 'ㅍ',
    13 : 'ㅎ',
    14 : 'ㅏ',
    15 : 'ㅑ',
    16 : 'ㅓ',
    17 : 'ㅕ',
    18 : 'ㅗ',
    19 : 'ㅛ',
    20 : 'ㅜ',
    21 : 'ㅡ',
    22 : 'ㅣ',
}

SINGLE_GESTURE = [
    [1, 1, 1, 1, 1, 1, 1, 1], # -
    [1, 1, 0, 0, 0, 1, 0, 0], # ㄱ
    [1, 1, 0, 0, 0, 1, 0, 1], # ㄴ
    [0, 1, 1, 0, 0, 1, 0, 1], # ㄷ
    [0, 1, 1, 1, 0, 1, 0, 1], # ㄹ
    [0, 0, 0, 0, 0, 0, 0, 0], # ㅁ
    [0, 1, 1, 1, 1, 0, 0, 0], # ㅂ
    [0, 1, 1, 0, 0, 1, 0, 0], # ㅅ
    [0, 0, 1, 1, 1, 0, 0, 0], # ㅇ
    [1, 1, 1, 0, 0, 1, 0, 0], # ㅈ
    [1, 1, 1, 1, 0, 1, 0, 0], # ㅊ
    [1, 0, 1, 0, 0, 1, 0, 0], # ㅋ
    [1, 0, 0, 0, 0, 0, 0, 0], # ㅎ
    [0, 1, 0, 0, 0, 0, 1, 0], # ㅏ
    [0, 1, 1, 0, 0, 0, 1, 0], # ㅑ,
    [0, 1, 0, 0, 0, 0, 1, 1], # ㅓ
    [0, 1, 1, 0, 0, 0, 1, 1], # ㅕ
    [0, 1, 0, 0, 0, 0, 1, 0], # ㅗ
    [0, 1, 1, 0, 0, 0, 1, 0], # ㅛ
    [1, 0, 0, 0, 0, 1, 0, 0], # ㅜ
    [0, 1, 0, 0, 0, 0, 1, 1], # ㅡ
    [0, 0, 0, 0, 1, 0, 0, 0], # ㅣ
]

SINGLE_CONSONANT = [
    0, # -
    0, # ㄱ
    0, # ㄴ
    0, # ㄷ
    0, # ㄹ
    0, # ㅁ
    0, # ㅂ
    0, # ㅅ
    0, # ㅇ
    0, # ㅈ
    0, # ㅊ
    0, # ㅋ
    0 # ㅎ
]

SINGLE_VOWEL = [
    0, # ㅏ
    0, # ㅑ
    0, # ㅓ
    0, # ㅕ
    0, # ㅗ
    0, # ㅛ
    0, # ㅜ
    0, # ㅡ
    0  # ㅣ
]

def make_data(infer_data):
    # x_min = 650.0
    # x_max = 4000.0
    # x_min = np.array([1300.,2700.,1850.,2150.,2150.])
    # x_max = np.array([3450.,3450.,3450.,3300.,3300.])
    x_min = np.array([1227., 1227., 1227., 1227., 1227., -179, -82, -177])
    x_max = np.array([4095., 4095., 4095., 4095., 4095., 179, 84, 179])

    gesture_data = np.array(infer_data, np.float)
    # normalize
    
    gesture_data = (gesture_data - x_min) / (x_max - x_min)

    gesture_data = gesture_data * 2.0 - 1.0
    
    if gesture_data.shape == (8,):
        gesture_data = gesture_data.reshape((1, 8))
            
    torch_data = torch.from_numpy(gesture_data)

    torch_data = torch.reshape(torch_data, (1, 1, 5, 8))

    # print(torch_data)

    return torch_data

def model_run(infer_data):

    torch_data = make_data(infer_data)

    model = ResNet()

    state_dict_path = "/home/jabblee/Desktop/CRC_collections/CRC_update/output/142_state_dict.pt"
    # state_dict_path = "/home/sun/Desktop/CRC/output/model1/28_state_dict_model.pt"

    model.load_state_dict(torch.load(state_dict_path))

    model = model.cuda()

    model.eval()

    torch_data = torch_data.cuda().float()

    pred = model(torch_data)

    # print("prediction : ", pred)

    class_idx = torch.argmax(pred, 1)

    return int(class_idx.item()), pred.cpu()

class RequestHandler(BaseHTTPRequestHandler):
    def __get_Post_Parameter(self, key):
        # 해당 클래스에 __post_param변수가 선언되었는지 확인한다.
        if hasattr(self,"_myHandler__post_param") == False:
            # 해더로 부터 formdata를 가져온다.
            data = self.rfile.read(int(self.headers['Content-Length']))
            if data is not None:
                self.__post_param = dict(urlparse.parse_qs(data.decode()))
            else :
                self.__post_param = {}
        if key in self.__post_param:

            return self.__post_param['start'][0], self.__post_param['end'][0]
        return None

    def __set_Header(self, code):
        self.send_response(code)
        self.send_header('Content-type','application/json')
        self.end_headers()

    # http 프로토콜의 body내용을 넣는다.
    def __set_response(self, data):
        global call_back_counter
        global valid_call_back_counter
        global data_list
        global pre_result
        global reset_flag
        global multi_motion_cnt
        global SINGLE_CONSONANT
        global SINGLE_VOWEL

        if (data):
            call_back_counter += 1

            #print("data : ", data)

            check_header = data[0].split(" ")[0]
            end_flag = data[1]

            
            # print(check_header)
            # print(end_flag)

            if (check_header == "#" and end_flag == "false"):
                
                test={"class":"-1", "endclass":"0"}
                self._send_class(test)
                
            elif (check_header=="*" or (check_header=="#" and end_flag == "true")):
                # x_max = np.array([3450.,3450.,3450.,3300.,3300.])
                # x_min = np.array([2800.,2850.,2800.,2800.,2800.])
                x_min = np.array([1227., 1227., 1227., 1227., 1227., -179, -82, -177])
                x_max = np.array([4095., 4095., 4095., 4095., 4095., 179, 84, 179])

                valid_call_back_counter += 1
                
                preprocess_data = re.findall(r'-?\d+', data[0])
                
                float_data = [float(i) for i in preprocess_data if i]
                
                if len(float_data) > 8:
                    print("------------------------------")
                    
                else:
                    data_list.append(float_data)
                    
                    if len(data_list) % 5 == 0:
                        class_idx, pred = model_run(data_list)
                        data_list = []
                        
                        print("class_idx : ", class_idx)

                        test = {"word": idx_to_label[class_idx], "endclass":"0"}
                        print(test["word"])
                        
                        self._send_class(test)
                        
            else:
                print("-------------------------------")         
                data_list = []
                
    def _send_class(self,dict):
        self.send_response(200)
        self.wfile.write(bytes(json.dumps(dict), "utf8"))


    def do_POST(self):
        self.__set_Header(200)
        self.__set_response(self.__get_Post_Parameter('start'))
        


if __name__=="__main__":
    httpd = HTTPServer((host, port), RequestHandler)
    print("Hosting Server on port 8080")
    httpd.serve_forever()