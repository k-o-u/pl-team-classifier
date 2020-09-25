import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

classes = ["千葉ロッテマリーンズ","福岡ソフトバンクホークス","東北楽天ゴールデンイーグルス","北海道日本ハムファイターズ", "埼玉西武ライオンズ", "オリックス・バファローズ"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model4.h5')#学習済みモデルをロードする

graph = tf.get_default_graph()

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template("index_html", answer="ファイルを選択してください！")
                # flash('ファイルがありません')
                # return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return render_template("index_html", answer="ファイルを選択してください！")
                # flash('ファイルがありません')
                # return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                #受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
                img = image.img_to_array(img)
                data = np.array([img])
                #変換したデータをモデルに渡して予測する
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_answer = "所属球団は " + classes[predicted] + " です"

                return render_template("index.html",answer=pred_answer, img_url=UPLOAD_FOLDER + '/' + filename)

        return render_template("index.html",answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)