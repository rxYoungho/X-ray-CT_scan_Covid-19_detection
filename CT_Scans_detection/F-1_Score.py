import sklearn.metrics as metrics
import numpy as np
import argparse
import time
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-y", "--yolo", required=True,
# 	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([args["yolo"], "yolov3_custom.names"])
labelsPath = "covid3.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "covid3_1000_last.weights"
# os.path.sep.join([args["yolo"], "yolov3_custom.weights"])
# configPath = os.path.sep.join([args["yolo"], "yolov3_custom.cfg"])
configPath = "covid_weight3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load Yolo
classes = []
with open("covid3.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

prediction = []
for count in range(1,1682):

    # load our input image and grab its spatial dimensions
    image = cv2.imread("COVID_CT_SCAN_3class/Mendaly/COVID2_CT/"+str(count)+".jpg")

    # print(image)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])
    try:
        for x in idxs.flatten():
            if LABELS[classIDs[x]] == "Covid-19":
                answer = 1 # Covid
            else:
                answer = 0
                print("틀림", answer)
            print(LABELS[classIDs[x]], answer)
        prediction.append(answer)
    except:
        pass


print(prediction)
print(len(prediction))
# k = np.zeros(len(prediction), dtype=int)
x = np.ones(len(prediction), dtype=int)
# y = np.concatenate((k, x), axis=None) #result

d = np.array(prediction) #나의 예측
# c = np.array(prediction_covid)
# p = np.concatenate((d, c), axis=None) #result

accuracy = np.mean(np.equal(x,d))
right = np.sum(x * d == 1)
precision = right / np.sum(d)
recall = right / np.sum(x)
f1 = 2 * precision*recall/(precision+recall)

print('accuracy',accuracy)
print('precision', precision)
print('recall', recall)
print('f1', f1)


print('accuracy', metrics.accuracy_score(x,d) )
print('precision', metrics.precision_score(x,d) )
print('recall', metrics.recall_score(x,d) )
print('f1', metrics.f1_score(x,d) )

print(metrics.classification_report(x,d))
print(metrics.confusion_matrix(x,d))