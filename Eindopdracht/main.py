from loadData import *
from model import *
from candidates import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json

def compute_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x = float(box1_area + box2_area - intersection_area)

    if x > 0: 
        overlap_ratio = intersection_area / x
    else:
        overlap_ratio = 0

    return overlap_ratio

def non_max_suppression(candidates, bboxes, overlap_thres):
    areas = []
    for bbox in bboxes:
        areas.append((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

    areas = np.array(areas)

    sorted_indices = np.argsort(-areas)

    sorted_bboxes = bboxes[sorted_indices]
    sorted_candidates = candidates[sorted_indices]

    sorted_bboxes_copy = np.copy(sorted_bboxes)
    sorted_candidates_copy = np.copy(sorted_candidates)

    picked_bboxes = []
    picked_candidates = []

    num_boxes = len(bboxes)

    for i in range(num_boxes):
        picked_bboxes.append(sorted_bboxes[i])
        picked_candidates.append(sorted_candidates[i])

        overlapping_boxes = []
        for j in range(i+1, num_boxes):
            if compute_overlap(sorted_bboxes_copy[i], sorted_bboxes_copy[j]) > overlap_thres:
                overlapping_boxes.append(j)

        sorted_bboxes_copy = np.delete(sorted_bboxes_copy, overlapping_boxes, axis=0)
        sorted_candidates_copy = np.delete(sorted_candidates_copy, overlapping_boxes, axis=0)

        num_boxes -= len(overlapping_boxes)

    return np.array(picked_candidates), np.array(picked_bboxes)

# Notes:
# Bij de predict, i.p.v. de index van de label op te zoeken op basis van de hoogste waarde ( meerdere classes ), dus softmax,
# wordt nu sigmoid gebruikt. Als de waarde dan dichter bij de label is is het dus wel een kentekenplaat en als die bij de 1 ligt niet.

# Elk plaatje croppen en opslaan als nieuw plaatje (file).
# for i in range(len(loadData()[0])):
#     saveCroppedImage( convertBBoxImage(cutOutBBox(i), (224,224)), 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/Plates/', 'plate_' + loadData()[0][i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

# file, xMin, xMax, yMin, yMax, label = loadData()
# for i in range(len(file)):
#     image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[i]) + ".jpg"))
#     image = cutOutRandom( image, (224, 224), xMin[i], xMax[i], yMin[i], yMax[i] )
#     saveCroppedImage( image, 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/No Plates/', 'no_plate_' + file[i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

# classifier = trainModel()

json_file = open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'r')
classifier = model_from_json(json_file.read(), custom_objects={'KerasLayer':hub.KerasLayer})
classifier.load_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')

classifier.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

vid = cv.VideoCapture(0)

while(True):

    ret, license_plate = vid.read()

    # license_plate = cv.imread('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Transfer Test/car3.jpg')
    # license_plate = cv.cvtColor(license_plate, cv.COLOR_BGR2RGB)

    # license_plate = cv.resize(license_plate, (1920,1080), cv.INTER_AREA)

    license_plate_candidates, bboxes = createCandidatePositions(license_plate)

    license_plate_candidates_NMS, bboxes_NMS = non_max_suppression(license_plate_candidates,bboxes,0.75)

    print(license_plate_candidates_NMS.shape)

    # license_plate_candidates, bboxes = sweepCandidates(createClustering(license_plate))

    # for i in range(len(license_plate_candidates)-1):
    #     print( license_plate_candidates[i].shape )
    #     print( bboxes[i] )
    #     cv.rectangle( license_plate, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]), (255,0,0), 3 )

    # print(license_plate_candidates[0].shape)

    convertedCandidates = []
    for i in range(len(license_plate_candidates_NMS)-1):
        convertedCandidates.append(convertBBoxImage(license_plate_candidates_NMS[i], (224,224)))

    license_plate_candidates_NMS = np.array(convertedCandidates)/255.0

    # print(license_plate_candidates.shape)

    results = classifier.predict(license_plate_candidates_NMS)

    # print(results)

    # results = []
    # bboxesResults = []
    # for i in range(len(license_plate_candidates)-1):
    #     print(license_plate_candidates[i].shape)
    #     license_plate_candidate = convertBBoxImage( license_plate_candidates[i], (224, 224) )
    #     license_plate_candidate = np.array(license_plate_candidate)/255.0
    #     print(license_plate_candidate.shape)
    #     results.append( classifier.predict(license_plate_candidate[np.newaxis, ...])[0] )
    #     bboxesResults.append( bboxes[i] )

    maxIndex = np.where(results == np.max(results))[0][0]
    print(maxIndex)
    print(results[maxIndex])
    if results[maxIndex] > 0.6:
        cv.rectangle( license_plate, (bboxes_NMS[maxIndex][0], bboxes_NMS[maxIndex][2]), (bboxes_NMS[maxIndex][1], bboxes_NMS[maxIndex][3]), (255,0,0), 3 )
    # break

    # for i in range(len(results)):
    #     if results[i][0] > 0.6:
    #         cv.rectangle( license_plate, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]), (255,0,0), 3 )
    #         print(results[i][0])

    # result = classifier.predict(license_plate_candidate[np.newaxis, ...])

    # print(result)
    # print(result.shape)

    # classifier.summary()

    # file, xMin, xMax, yMin, yMax, label = loadData()
    # image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[0]) + ".jpg"))

    # image = cutOutRandom( image, (224, 224), xMin[0], xMax[0], yMin[0], yMax[0] )
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(license_plate)
    # plt.show()
    cv.imshow('frame', license_plate)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()