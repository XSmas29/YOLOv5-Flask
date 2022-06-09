import torch
import numpy as np
import cv2
import easyocr
import argparse
lang = easyocr.Reader(['en', 'id'])
ocr_th = 0.3

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

def deteksi(frame, model):
    frame = [frame]
    print(f"[INFO] Loading Sedang Mendeteksi Teks Pada Gambar")
    hasil = model(frame)

    labels, coordinates = hasil.xyxyn[0][:, -1], hasil.xyxyn[0][:, :-1]
    return labels, coordinates

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    teks = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            teks.append(result[1])
    return teks

def recognize_text(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nteks = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    ocr_result = reader.readtext(nteks)
    print(ocr_result)
    text = filter_text(region=nteks, ocr_result=ocr_result, region_threshold= region_threshold)
    if len(text) == 1:
        text = text[0].upper()
    return text

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total Teks {n} yang terdeteksi pada gambar. . . ")
    print(f"[INFO] Melakukan pengulangan pada teks yang terdeteksi. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.4: ### threshold value for detection. We are discarding everything below this value
            #print(f"[INFO] Ektrasi Bounding Box Koordinat. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            coords = [x1,y1,x2,y2]
            text_recog = recognize_text(img = frame, coords= coords, reader= lang, region_threshold= ocr_th)
            #print(text_recog)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{text_recog}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
    return frame

if __name__ == "__main__":
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model = torch.hub.load('.', 'custom', path='yolov5sV2_epoch_56.pt', source='local') ### The repo is stored locally

    classes = model.names ### class names in string format
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = deteksi(frame, model = model) ### DETECTION HAPPENING HERE    
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame = plot_boxes(results, frame)

    print(f"[INFO] Cleaning up. . . ")
    cv2.destroyAllWindows()