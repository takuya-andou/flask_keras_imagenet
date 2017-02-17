# -*- coding=utf-8 -*-
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import os
import numpy as np
import sys
from flask import json
from flask import Response
from flask import jsonify
from flask import Flask, request, redirect, url_for,make_response, current_app
from datetime import timedelta
from functools import update_wrapper

from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 学習済みのVGG16をロード
# 構造とともに学習済みの重みも読み込まれる
model = VGG16(weights='imagenet')
# model.summary()



# 
# Access-Control-Allow-Originの対策処理
# 同一サーバーで呼び出す場合？は不要なのでコメントアウトしておく
# 

# # Access-Control-Allow-Originのエラーが出たのでコピペ
# # これで正しい処理なのかは不明
# def crossdomain(origin=None, methods=None, headers=None,
#                 max_age=21600, attach_to_all=True,
#                 automatic_options=True):
#     if methods is not None:
#         methods = ', '.join(sorted(x.upper() for x in methods))
#     if headers is not None and not isinstance(headers, basestring):
#         headers = ', '.join(x.upper() for x in headers)
#     if not isinstance(origin, basestring):
#         origin = ', '.join(origin)
#     if isinstance(max_age, timedelta):
#         max_age = max_age.total_seconds()

#     def get_methods():
#         if methods is not None:
#             return methods

#         options_resp = current_app.make_default_options_response()
#         return options_resp.headers['allow']

#     def decorator(f):
#         def wrapped_function(*args, **kwargs):
#             if automatic_options and request.method == 'OPTIONS':
#                 resp = current_app.make_default_options_response()
#             else:
#                 resp = make_response(f(*args, **kwargs))
#             if not attach_to_all and request.method != 'OPTIONS':
#                 return resp

#             h = resp.headers

#             h['Access-Control-Allow-Origin'] = origin
#             h['Access-Control-Allow-Methods'] = get_methods()
#             h['Access-Control-Max-Age'] = str(max_age)
#             if headers is not None:
#                 h['Access-Control-Allow-Headers'] = headers
#             return resp

#         f.provide_automatic_options = False
#         return update_wrapper(wrapped_function, f)
#     return decorator

# # Access-Control-Allow-Originのエラーが出たのでコピペ
# # これで正しい処理なのかは不明
# def show_events():
#    response.headers['Access-Control-Allow-Origin'] = '*'
#    return response


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@crossdomain(origin='*')
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return "no file"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return "file name is null"

        # ファイルが正しくアップロードされた場合、保存してAIの処理へ
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


            # 
            # ここからがAIの処理
            # 
            inputfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 引数で指定した画像ファイルを読み込む
            # サイズはVGG16のデフォルトである224x224にリサイズされる
            img = image.load_img(inputfilename, target_size=(224, 224))

            # 読み込んだPIL形式の画像をarrayに変換
            x = image.img_to_array(img)

            # 3次元テンソル（rows, cols, channels) を
            # 4次元テンソル (samples, rows, cols, channels) に変換
            # 入力画像は1枚なのでsamples=1でよい
            x = np.expand_dims(x, axis=0)

            # Top1のクラス名だけを返す
            # VGG16の1000クラスはdecode_predictions()で文字列に変換される
            preds = model.predict(preprocess_input(x))
            results = decode_predictions(preds, top=1)[0]
            result_string = ""
            for result in results:
                return result[1]

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''



''' Flask起動 '''
if __name__ == '__main__':
    '''already in use? '''
    app.run(debug=True, host='0.0.0.0', port=9090)
