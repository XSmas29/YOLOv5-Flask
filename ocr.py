import torch
import numpy as np
import cv2
import keras_ocr
import argparse
#lang = easyocr.Reader(['en', 'id'])
ocr_th = 0.3
recognizer = keras_ocr.recognition.Recognizer(
    weights='kurapan'
)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

def deteksi(frame, detector):
    #mendeteksi lokasi teks
    frame = [frame]
    print(f"[INFO] Loading Sedang Mendeteksi Teks Pada Gambar")
    hasil = detector(frame)

    coordinates = hasil.xyxyn[0][:, :-1]
    return coordinates

def recognize_text(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nteks = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    ocr_result = (coords, recognizer.recognize(nteks))
    print(ocr_result)
    return ocr_result

def plot_boxes(results, frame):
    cord = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total Teks {cord.size()} yang terdeteksi pada gambar. . . ")
    print(f"[INFO] Melakukan pengulangan pada teks yang terdeteksi. . . ")
    ### looping through the detections
    for i, xyxy in enumerate(cord):
        row = xyxy
        if row[4] >= 0.4: ### threshold value for detection. We are discarding everything below this value
            #print(f"[INFO] Ektrasi Bounding Box Koordinat. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            coords = [x1,y1,x2,y2]
            recognize_text(img=frame, coords=coords)
    return frame

if __name__ == "__main__":
    print(f"[INFO] Loading detector... ")
    ## loading the custom trained detector
    # detector =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    detector = torch.hub.load('.', 'custom', path='yolov5sV2_epoch_56.pt', source='local') ### The repo is stored locally

    classes = detector.names ### class names in string format
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = deteksi(frame, detector = detector) ### DETECTION HAPPENING HERE
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame = plot_boxes(results, frame)

    print(f"[INFO] Cleaning up. . . ")
    cv2.destroyAllWindows()