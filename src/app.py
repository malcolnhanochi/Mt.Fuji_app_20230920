# 必要なモジュールのインポート
from IPython.core.debugger import py3compat
from torchvision import transforms
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
import torch
import pandas as pd
import os
from torchvision.models import resnet18
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt



# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(500),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True) 
        self.fc = nn.Linear(1000, 2)
        # self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        #学習時に使ったのと同じ順伝播
        # h = self.bn(x)
        h = self.feature(x)
        h = self.fc(h)
        return h





















# from MtFuji import transform, Net # MtFuji.py から前処理とネットワークの定義を読み込み
# import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（weights.pt）を読み込み
    net.load_state_dict(torch.load('./weights.pt', map_location=torch.device('cpu')))    
    # net.load_state_dict(torch.load(r'src\weights.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

#　推論したラベルから富士山が実際のかAIのかを返す関数
def getName(label):
    if label==0:
        return 'AIが合成した富士山の画像'
    elif label==1:
        return '実際の富士山の写真'

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            MtFujiName_ = getName(pred)
            return render_template('result.html', MtFujiName=MtFujiName_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)