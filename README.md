# ***CloudPhysician***
# ***THE VITAL EXTRACTION CHALLENGE***

### ***Extracting Patient Vitals from the given image of vital sign monitor using ML model, Image Processing and Optical Character Recognition.***

## ***About the data***

- The data consists of various images of health monitors. The classification dataset of the 4 types of monitors was used. Python code was used to randomly shuffle the images and pick 800 for training, and 200 for validation.
The script created the following folder format: 
dataset - images - (train + val folders)
        - labels - (train + val folders)
All the image and label filenames are indexed by numbers. The various label files contain annotations for the images. An example annotation is:
```
0 0.78359375 0.14375 0.065625 0.09583333333333334
3 0.89921875 0.25416666666666665 0.0453125 0.06944444444444445
4 0.794140625 0.5027777777777778 0.08515625 0.10277777777777777
5 0.778515625 0.6208333333333333 0.05546875 0.10277777777777777
6 0.34765625 0.37083333333333335 0.678125 0.09166666666666666
7 0.34765625 0.6201388888888889 0.6734375 0.08472222222222223
8 0.348046875 0.5090277777777777 0.67578125 0.10694444444444444
```

- The first index is the label, and the 4 indices that follow are the normalized bounding box coordinates.
The labels correspond to the indices of the matrix `['HR','SBP','DBP','MAP','SPO2','RR','HR_W','RR_W','SPO2_W']`

- A dataset.yaml file is used to configure the model. It's format is:

```
path: dataset
train: images/train
val: images/val

nc: 9

names: `['HR','SBP','DBP','MAP','SPO2','RR','HR_W','RR_W','SPO2_W']`

```
## ***Getting Started***

To use the model, just run the corresponding cells in the Jupyter notebook. The cells will download the pretrained model from GitHub, and use it to generate the bounding boxes which will later be used for OCR detection.

## ***Neural Network Architecture***

The bounding boxes will also be generated using the yolov5 model:



***Why this architecture?***

YOLOv5 (You Only Look Once version 5) is an object detection model used in computer vision tasks. It is a fast and accurate model that can perform object detection in real-time. YOLOv5 is the latest version in the YOLO series and is based on a single shot multi-box detection (SSD) architecture. It uses anchor boxes to predict bounding boxes for objects in an image and assigns a class label to each bounding box. YOLOv5 is known for its ability to detect objects in real-time and its relatively low computational requirements compared to other object detection models.





Other architectures like the Faster-RCNN and SSD were tried, but the YoloV5 model was found to give the best (and fastest) results. Yolov5 allows us to use a state of the art architecture to quickly detect objects. The model was trained for the problem specification by using the custom dataset. 

## ***Training the model***
The model was trained for 30 epochs and the following table was obtained:


|Class   |  Images | Instances  |     P    |      R    |  mAP50   |  mAP50-95: 100% 10/10  |
| ------ | ------- | ---------- | -------- | --------- | -------- | ---------------------- |
|all     |   300   |    2161    |  0.783   |   0.957   |   0.837  |      0.669             |
|HR      |   300   |     300    |  0.985   |       1   |   0.995  |      0.886             |
|SBP     |   300   |     273    |      1   |   0.867   |   0.989  |      0.842             |
|DBP     |   300   |     269    |  0.757   |       1   |   0.985  |      0.816             |
|MAP     |   300   |     271    |   0.98   |       1   |   0.995  |      0.718             |
|SPO2    |   300   |     281    |  0.762   |       1   |   0.988  |       0.88             |
|RR      |   300   |     298    |  0.988   |   0.835   |    0.98  |      0.829             |
|HR_W    |   300   |     163    |  0.539   |   0.994   |    0.54  |      0.379             |
|RR_W    |   300   |     160    |  0.541   |   0.988   |   0.573  |      0.342             |
|SPO2_W  |   300   |     146    |  0.492   |   0.932   |   0.494  |       0.33             |



## ***Output of ML Model***

Now, the best model was used to create bounding boxes for the various classes:<br>

It outputs two matrices:
- 1 dimensional matrix containing the labels of the predicted classes
- 2 dimensional matrix, containing the bounding box coordinates (normalised x and y, width and height) and confidence for the predicted labels
The predicted bounding boxes can also be displayed using `results.show()`

The two matrices are processed and used later for OCR detection.

## ***Processing of Image***

- Using the bounding box coordinates image is cropped.
- Cropped Image is then denoised using several denoising techniques and thresholded.
- The threshold value chosen is the optimal value where we can binarize the image without losing the text.
- Processed Image is then passed through OCR to predict the text in the image.
- All the predicted value is reported in the form of python dictionary format.

## ***OCR Configuration***

```
r"--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789/"
```

This is a command-line argument for the Tesseract OCR engine. The arguments are as follows:

- `--psm 8` : Page Segmentation Mode 8 is used to recognize a single digit as an instance of the character class.

- `--oem 3` : The OCR Engine Mode 3 is used, which is a deep learning based recognition system.

- `-c tessedit_char_whitelist=0123456789/` : The character whitelist is set to the numbers 0 to 9, allowing the OCR engine to recognize only these characters.

The command line options are used to configure the behavior of the Tesseract OCR engine for optimal recognition of a specific type of text, in this case, single digits.

## ***HR Wave Digitization***

- The YoloV5 model is predicting the HR Wave and returning the coordinates of the bounding box.
- After that the HR bounding box is cropped to give an image containing the HR for digital ECG analysis.
- The function `digitalECG(path)` first reads an image, scales it, performs noise reduction and thresholding on it to obtain a binary image.
- Then it finds the R peaks in the ECG signal by computing the mean, mode, and R_central_difference of the digitized ECG signal.
- Finally, it plots the digital ECG signal, the thresholded image, and the original image.

# ***Explaining Functions Used***

## ***cropImage(img, normalized_coordinates)*** 
- This function crops an image (2D numpy array `img`) using the normalized coordinates specified in `normalized_coordinates` (a list of 4 values, each representing a fraction of the total image size).
- The function first calculates the width and height of the image using `img.shape[1]` and `img.shape[0]`, respectively.
- Then it extracts a rectangular region of the image, defined by the top-left and bottom-right corners, which are computed by scaling the normalized coordinates by the image size.
- The cropped region is returned as a new 2D numpy array.

## ***crop_Image(img, normalized_coordinates)***
- It is function similar to `cropImage`. It is specifically used to crop image of vital values by increasing the margin of cropped image so that edges of the numbers are not cut and is recognized perfectly for OCR.

## ***text_r(img)***
It is a function that performs optical character recognition (OCR) on an input image "img". The function does the following steps:

- Defines the OCR configuration using the "my_config" variable, which specifies that only digits and "/" should be recognized, and sets the Page Segmentation Mode (psm) to 8 and the OCR Engine Mode (oem) to 3.

- Removes noise from the image using the fastNlMeansDenoisingColored function from OpenCV library.
  
- Converts the image to binary format using the threshold function from OpenCV library.

- Converts the binary image to grayscale using the cvtColor function from OpenCV library.

- Uses the image_to_string function from the Tesseract OCR library to perform OCR on the grayscale image, and returns the recognized text.

## ***hr_wave_detection(img, labels, coords)***
This function takes an image, an array of labels and an array of coordinates as inputs. The function performs the following steps:

- Loops over each label in the input "labels" array.
- If the label is equal to 6, it calls the "crop_image" function to crop the image using the corresponding coordinates in the "coords" array.
- Calls the "cv2_imshow" function to display the cropped image, which is supposed to be the HR waveform.

## ***ocr(img, complete_coords, labels)***
- This function performs optical character recognition (OCR) on an input image `img` using `complete_coords` and `labels` as parameters.
- It creates an output dictionary `output_dict` and uses the reference list `reference` which has labels for the extracted data.
- The function crops the input image using the `crop_image` function and performs text recognition on the cropped image using the `text_r` function.
- The extracted text is stored in the `output_dict` with its corresponding label from the `reference` list.
- The function then returns the `output_dict` after all 6 labels have been processed.

## ***inference(image_path:str)***
It is a function that takes an image file path as an input and returns the result of OCR. The function performs the following steps:

- Calls the "monitor_model" function on the input image to get the locations of text regions in the image.
- Calls the "cropImage" function to crop the image based on the locations of the text regions.
- Calls the "vitals_model" function on the cropped image to get updated locations of the text regions.
- Calls the "ocr" function on the cropped image and the updated text region locations to perform OCR on the image.
- Returns the result of OCR.
