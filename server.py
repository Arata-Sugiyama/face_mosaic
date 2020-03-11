from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2

from datetime import datetime
import os
import string
import random

SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__, static_url_path="")

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route('/')
def index():
    return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1])

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

# 参考: https://qiita.com/yuuuu3/items/6e4206fdc8c83747544b
@app.route('/upload', methods=['POST'])
def upload():
    if request.files['image'].filename != u'':
        # 画像として読み込み
        stream = request.files['image'].read()
        img_array = np.fromstring(stream, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cascade_file = "./haarcascade_frontalface_alt.xml"
        cascade = cv2.CascadeClassifier(cascade_file)
        face_list = cascade.detectMultiScale(img,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(100,100))
        mosaic_rate = 30
        # 変換
        for (x,y,w,h) in face_list:
            face_img = img[y:y+h,x:x+w]
            face_img = cv2.resize(face_img,(w//mosaic_rate,h//mosaic_rate))
            face_img = cv2.resize(face_img,(w,h),
                              interpolation=cv2.INTER_AREA)

            img[y:y+h,x:x+w] = face_img


        # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
        save_path = os.path.join(SAVE_DIR, dt_now + ".png")
        
        cv2.imwrite(save_path, img)

        print("save", save_path)

        return redirect('/')

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8888)
