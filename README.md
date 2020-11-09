# X-ray and CT scan COVID-19 Detection using Machine Learning Techniques

# Problem Statement

  35,248,330, which is almost half of South Korea’s population, is the total number of worldwide coronavirus cases as of October 4th. 1,039,541 people have died, and 26,225,235 have recovered after contracting the virus. The number of active coronavirus patients is still 7,983,554, and 66,267 have progressed to a more severe state. 

  The coronavirus disease (COVID-19) originated from Wuhan, China in December 2019. Previous cases of coronavirus outbreaks include Middle East respiratory syndrome (MERS-CoV) and severe acute respiratory syndrome (SARS-CoV). Past examples of automated chest radiography (X-ray) diagnoses are not as easily accessible as current cases due to the comparatively little attention the outbreaks received, and the subsequent lack of interest in the radiography data.

  Recently, many hospitals have begun to realize the necessity of automated COVID-19 detection due to the rapid growth in the number of COVID-19 patients. Radiography physicians and doctors were able to differentiate between pneumonia-infected and healthy lungs through patients’ chest X-ray images. Now, however, doctors may have to recognize the difference between COVID-19 and regular pneumonia from X-ray images alone. The problem here is that the symptoms shown in the X-ray image of COVID-19 and pneumonia are similar enough that a regular doctor cannot easily differentiate the two. Specialized radiologists can diagnose COVID-19 more easily, but there are far fewer specialists than regular doctors.

  There are two solutions to this problem. The first solution is providing a machine learning classification model to do the differentiation. With less work and involvement from the doctors, the process of accurately diagnosing the patient will become faster. In the cases where the machine learning model fails to provide high confidence in diagnosing, a possible secondary solution would be to use CT scans instead. Using CT scans may provide higher accuracy in classifying COVID-19 compared to X-ray images due to the higher resolution of the produced scans [4]. However, CT scans and machines are both more time-consuming and less common than their X-ray counterparts. It is difficult to provide CT scans for all the patients, which is why it would better serve a secondary role. It is also important to note that doctors find it difficult to recognize COVID-19 from visual inspection of CT scans, which is why machine learning is needed to corroborate the diagnosis.

# COVID-19 Detection model (X-ray)

  Our initial model focused on classifying 3 classes with a resnet model, outputting the argmax of the classes. This resulted in an F1 score of 93.97 in average of 3 classes. Our new model now focuses on tackling 3 distinct Resnet Models, where each Resnet model handles the classes as displayed below.

![Image of the model](https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/blob/master/img/resnet.png)

  Additionally, the following is a visualization of the model made through Tensorflow:

![Tri-Resnet model](https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/blob/master/img/tri-resnet.png)

# COVID-19 Detection model (CT Scans)

  By using YOLO: real time object detection tool, the model has gained new knowledge of two different data: Normal chest CT scan and COVID-19 chest CT scan. In order to verify the accuracy of YOLO darknet (one of the pre-trained neural networks) when used in a medical area, it was firstly trained as a binary classification model. The created model gets one input of CT scan image from the end-user, and it provides the confidence, accuracy, and the class that the input is involved in. 
  
  When tested using cross validation, from 100 images out of 1000 images, the F-1 score was about 95% when detecting the Normal chest CT scan imag	es. The F-1 score of COVID-19 chest CT scan images was about 99%. Both had high accuracy of detecting the features since the segmentation method was used before they were trained (pre-processing), and the method also was used in the input data.

```
@COVID-19 Detection
              precision      recall      f1-score   support

           0       0.00        0.00          0.00         0
           1       1.00        0.91          0.95        97

    accuracy                                 0.91        97
   macro avg       0.50        0.45          0.48        97
weighted avg       1.00        0.91          0.95        97

@Normal Detection
              precision    recall  f1-score   support

          0       1.00      0.98      0.99        99
          1       0.00      0.00      0.00         0

    accuracy                          0.98        99
   macro avg       0.50      0.49     0.49        99
weighted avg       1.00      0.98     0.99        99

```

  The model was trained with 100 representative chest CT scan images for each Normal and COVID-19 class. The model, however, showed both low accuracy and F1-score when faced with unseen data. Below is the result of 10,000 unseen data testing.

```
Unseen Data [F1-Score of both Normal and COVID-19 class]
              precision    recall  f1-score   support

           0       0.75      0.80      0.77      9305
           1       0.40      0.35      0.37      3674

    accuracy                           0.67     12979
   macro avg       0.58      0.57      0.57     12979
weighted avg       0.65      0.67      0.66     12979

```
  In order to improve the accuracy of the current model, more dataset will be trained to the model, and when YOLO can detect the COVID-19 in a high accuracy, the model will be rebuilt as a multiple classification (Normal, Pneumonia, COVID-19) model soon.

# Data pre-processing

Before Segmentation             |  After Segmentation
:-------------------------:|:-------------------------:
![Image of the before](https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/blob/master/img/before.png)  |  ![Image of the after](https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/blob/master/img/after.png)


# Detection Result (CT)
![Image of the model](https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/blob/master/img/normal_test.png)

  

# Operating System
```
All of the codes were written in the environment of Windows 10.
```

# Bug Tracking

#### Any users can submit their bug report in "Github Issues" of our repository. 
#### Check out the recent issues from : https://github.com/rxYoungho/X-ray-CT_scan_Covid-19_detection/issues

# Dataset

#### https://www.kaggle.com/anaselmasry/ct-images-for-covid-normal-pneumonia-mendeley
#### https://drive.google.com/drive/folders/1LC_l4sVvoUgg2MsZ8veoeB8OY6v_vNiq?usp=sharing 
#### X-ray Dataset that is merged from public sources from: COVID19 Radiography Database, Figure1 COVID Chest X-ray Dataset, COVID19 Detection X-ray Dataset, Detecting COVID19 in X-ray Images Dataset, COVIDX Dataset
# Contribution
-------------------
### Daekyung Kim
###	Jey Kang
###	Youngho Kim


