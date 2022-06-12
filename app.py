from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
import ast
import json

app = Flask(__name__)
app.debug = True
uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/detect", methods=['POST'])
def detect():
    if not request.method == "POST":
        return
    video = request.files['video']
    video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    print(video)
    output = subprocess.run(['python', 'ocr.py','--image', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True, stdout=subprocess.PIPE) #
    lines = output.stdout.splitlines()
    result = []
    for line in lines:
        if not line.startswith(b"[INFO]"):
            strline = str(line, "utf-8").strip("][")
            if strline != '':
                tuple = list(ast.literal_eval(strline))
                print(tuple)
                result.append(tuple)
    return json.dumps(result)

    # subprocess.run(['python', 'detect.py', '--hide-conf', '--hide-labels', '--conf-thres', '0.4', '--save-crop','--save-txt', '--weights', 'yolov5sV2_epoch_56.pt', '--source', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True)
    obj = secure_filename(video.filename)
    return obj

if __name__ == "__main__":
    app.run(debug=True)
# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static/video_1.mp4', code=200))