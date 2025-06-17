# Yolo-v8-Driver-License

Driver License OCR using YOLOv8 and TrOCR
This project aims to automate the extraction of key information from US driver licenses using a combination of computer vision models, significantly reducing manual data entry and potential errors.

## Table of Contents
- Introduction
- Features
- How It Works
- Training the YOLOv8 Model
- Model Application
- Results
- Limitations
- Getting Started
    - Prerequisites
    - Installation
    - Usage

## Introduction
Manually transcribing information from driver licenses is a time-consuming and error-prone process. This project offers an automated solution by leveraging advanced deep learning models to accurately identify and extract relevant data points such as name, address, date of birth, license number, and more. By digitizing this process, we can achieve significant time savings and improve data accuracy.

## Features
- Automated Data Extraction: Automatically extracts key information from US driver licenses.
- YOLOv8 for Object Detection: Utilizes YOLOv8 for precise localization of information fields on the driver's license.
- TrOCR for Optical Character Recognition: Employs TrOCR for robust character recognition within the identified fields.
- End-to-End Solution: From image input to structured DataFrame output.
- Error Reduction: Minimizes human errors associated with manual data entry.

## How It Works
The project follows a multi-stage approach to extract information:

1. Object Detection (YOLOv8): A YOLOv8 model, trained on a custom dataset of demo US driver licenses, identifies and localizes key information fields (e.g., "Address", "DOB", "License Number") on an input driver's license image.
2. Image Preprocessing (Rotation & Classification): Auxiliary models are used to ensure the image is correctly oriented and potentially classified for optimal OCR performance.
3. Optical Character Recognition (TrOCR): For each identified field, the TrOCR model transcribes the text, converting the image region into readable characters.
4. Data Structuring: The extracted text information is then compiled into a structured Pandas DataFrame for easy access and further processing.

## Training the YOLOv8 Model
The YOLOv8 model is trained on a custom dataset of demo US driver licenses. This dataset includes bounding box annotations for the following key information fields:

- Address
- Class
- Date of Birth (DOB)
- Expiration Date (Exp date)
- First name
- Issue date
- Last name
- License number
- Sex
- State
You can train your own YOLOv8 model using the following command:
```
python train.py --img 640 --batch 16 --epochs 50 --data /content/drive/MyDrive/driver_license_extraction/dataset.yaml --weights yolov8s.pt
```

This training process will generate a .pt model file (e.g., best.pt) containing the trained YOLOv8 weights.


## Model Application
Once the YOLOv8 model is trained, and other necessary models are available, the system can be used to extract information from real driver's license images. The core logic for loading and applying the models looks like this:

```
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO # Assuming ultralytics library for YOLO

# Load processor and TrOCR model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trocr_model.to(device)

# Load other pre-trained models (assuming they are in .pt format)
rotate_model = torch.load(r'/content/drive/MyDrive/Driving_License_OCR-main-v2/resnet/rotate_entire_2.pt', map_location=device)
class_model = torch.load(r'/content/drive/MyDrive/Driving_License_OCR-main-v2/resnet/class_entire_2.pt', map_location=device)
model = YOLO(r'/content/drive/MyDrive/Driving_License_OCR-main-v2/weights/best.pt') # Your trained YOLOv8 model
state_model = torch.load(r'/content/drive/MyDrive/Driving_License_OCR-main-v2/resnet/state_class_2.pt', map_location=device)

# ... (further code to apply models to input images and extract data)
```
After processing, the extracted data for each driver's license will be organized into a Pandas DataFrame.

## Results
The primary output of this project is a structured DataFrame containing the extracted information for each processed driver's license. This allows for easy integration with other systems or for data analysis.

## Limitations
While this project offers significant advantages, it's important to acknowledge its current limitations:

Accuracy with Unclear Images: The accuracy of the extracted information heavily depends on the clarity and quality of the input image. Blurry, low-resolution, or poorly lit images may lead to decreased accuracy.
Generalizability: The model is primarily trained on demo US driver licenses. Its performance on driver licenses from other countries or significantly different layouts within the US might be suboptimal without further training.
Model Imperfections: Despite advanced techniques, the models are not flawless and may occasionally misinterpret characters or fields.
## Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
- Python 3.8+
- pip

