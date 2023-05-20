'''
依赖：pip install ai-hub #(version>=0.3.5)
测试用例：
model为y=2*x
请求数据为json:{"img":3}
-----------
post请求：
curl localhost:8080/tccapi -X POST -d '{"img":3}'
返回结果 6
'''
from ai_hub import inferServer
import json
import os
import shutil
import base64
from io import BytesIO
from PIL import Image


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        print("init_myInfer")

    # 数据输入elem: dict结构,表示某一特定时刻的Wind/Radar/Precip数据
    # elem["Wind"]: {
    #   "wind_001.png": "base64的png图片内容",
    #   "wind_002.png": "base64的png图片内容",
    #   ...
    #   "wind_020.png": "base64的png图片内容",
    # }
    # elem["Radar"]: {
    #   "radar_001.png": "base64的png图片内容",
    #   "radar_002.png": "base64的png图片内容",
    #   ...
    #   "radar_020.png": "base64的png图片内容",
    # }
    # elem["Precip"]: {
    #   "precip_001.png": "base64的png图片内容",
    #   "precip_002.png": "base64的png图片内容",
    #   ...
    #   "precip_020.png": "base64的png图片内容",
    # }

    # 数据前处理
    # 本示例的做法是将天池服务器传入的特定时刻的Wind/Radar/Precip数据写入到本地图片，供选手参考。
    # 选手可以根据自己的处理逻辑进行数据预处理后return给pridect(data)。
    def pre_process(self, data):
        print("pre_process")
        data = data.get_data()

        # json process
        json_data = json.loads(data.decode('utf-8'))

        # 将图片写到本地的submit目录：
        if os.path.exists('submit'):
            shutil.rmtree('submit')
            print ('Delete submit folder')
            os.makedirs('submit')
            print ('Create submit folder')

        for category in ['Wind', 'Precip', 'Radar']:
            category_path = os.path.join('submit', category)
            os.makedirs(category_path)
            for png_name, base64_string in json_data[category].items():
                ## 请选手注意，在天池的流评测服务器环境下，此处获取到的base64_string其实是一个list结构，非str，选手需要显性得取第一个元素才能得到正确的图片base64编码内容
                base64_string = base64_string[0]
                file_name = os.path.join(category_path, png_name)
                img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_string)))
                img.save(file_name)
                #print ('success save ', file_name)

        return data

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写
    def predict(self, data):
       ret = self.model(data)
       return ret

    # 数据后处理
    def post_process(self, data):
        print("post_process")

        #此处示例仅演示生成返回格式，未经过predict（data）;直接读取本地图片。选手可根据实际逻辑实现。
        # 与pre_process类似，将模型预测后的图片文件，以base64的方式返回，返回字段见如下的定义：
        # elem["Wind"]: {
        #   "wind_001.png": "base64的png图片内容",
        #   "wind_002.png": "base64的png图片内容",
        #   ...
        #   "wind_020.png": "base64的png图片内容",
        # }
        # elem["Radar"]: {
        #   "radar_001.png": "base64的png图片内容",
        #   "radar_002.png": "base64的png图片内容",
        #   ...
        #   "radar_020.png": "base64的png图片内容",
        # }
        # elem["Precip"]: {
        #   "precip_001.png": "base64的png图片内容",
        #   "precip_002.png": "base64的png图片内容",
        #   ...
        #   "precip_020.png": "base64的png图片内容",
        # }
        elem = {}
        # 读取模型预测好的文件，按照如下格式返回, 本例为了说明问题，直接去读输入文件的第一条作为输出结果：
        elem['Wind'] = {}
        elem['Radar'] = {}
        elem['Precip'] = {}
        for category in ['Wind', 'Precip', 'Radar']:
            mock_file_path = os.path.join('submit', category, f"{category.lower()}_001.png")
            print ('mock_file_path: ', mock_file_path)
            binary_content = open(mock_file_path, 'rb').read()
            base64_bytes = base64.b64encode(binary_content)
            base64_string = base64_bytes.decode('utf-8')
            for idx in range(1, 21):
                file_name = os.path.join('', f"{category.lower()}_{idx:03d}.png") 
                print ('Post_process: ', category, file_name)
                elem[category][file_name] = base64_string
       
        # 返回json格式
        return json.dumps(elem)


if __name__ == "__main__":
    mymodel = lambda x: x * 2
    my_infer = myInfer(mymodel)
    my_infer.run(debuge=True)#, nohug=False)  # 默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息
    # ！！！nohug参数默认为False,第二阶段记得修改为True提交，即可持久化inferServer服务，等待正式测评
