from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
import ast
import json
import torch
import numpy as np
import cv2
import keras_ocr
import argparse

app = Flask(__name__)
app.debug = True
uploads_dir = os.path.join(app.instance_path, 'uploads')

ocr_th = 0.3
recognizer = keras_ocr.recognition.Recognizer(
    weights='kurapan'
)

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
    # output = subprocess.run(['python', 'ocr.py','--image', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True, stdout=subprocess.PIPE) #
    # lines = output.stdout.splitlines()
    # result = []
    # for line in lines:
    #     if not line.startswith(b"[INFO]"):
    #         if line.endswith(b")"):
    #             print(line)
    #             strline = str(line, "utf-8")
    #             if strline != '':
    #                 tuple = list(ast.literal_eval(strline))
    #                 result.append(tuple)
    # return json.dumps(result)
    # subprocess.run(['python', 'detect.py', '--hide-conf', '--hide-labels', '--conf-thres', '0.40', '--save-crop','--save-txt', '--weights', 'yolov5sV2_epoch_69.pt', '--source', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True)
    # obj = secure_filename(video.filename)
    # return obj
    list_result = OCR(os.path.join(uploads_dir, secure_filename(video.filename)))
    print("result:")
    json_result = jsonify(
        {
            "prediction": list_result
        }
    )
    print(json_result)
    return json_result

def OCR(image):
    image = cv2.imread(image)
    detector = torch.hub.load('.', 'custom', path='yolov5sV2_epoch_69.pt', source='local')
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #deteksi lokasi textnya
    results = get_coordinates(frame, detector)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    #deteksi value textnya
    list_result = recognize_text(results, frame)
    return list_result

def get_coordinates(frame, detector):
    #mendeteksi lokasi teks
    frame = [frame]
    hasil = detector(frame)
    coordinates = hasil.xyxyn[0][:, :-1]
    return coordinates

def recognize_text(results, frame):
    cord = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    # print(f"[INFO] Total Teks {cord.size()} yang terdeteksi pada gambar. . . ")
    # print(f"[INFO] Melakukan pengulangan pada teks yang terdeteksi. . . ")
    list_result = []
    ### looping through the detections
    for i, xyxy in enumerate(cord):
        row = xyxy
        if row[4] >= 0.375: ### threshold value for detection. We are discarding everything below this value
            #print(f"[INFO] Ektrasi Bounding Box Koordinat. . . ")
            xmin, ymin, xmax, ymax = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            coords = [xmin, ymin, xmax, ymax]
            nteks = frame[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
            ocr_result = (coords, recognizer.recognize(nteks))
            print(ocr_result)
            list_result.append(ocr_result)
    return list_result

if __name__ == "__main__":
    app.run(debug=True)
# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static/video_1.mp4', code=200))