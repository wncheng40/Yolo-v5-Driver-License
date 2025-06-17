import os
import pandas as pd
from PIL import Image
import torch
from preprocessing import get_position, get_type, rotation


def read_img_old(dl_image, model, processor, trocr_model):
    """Detects labels on a Driver's License (DL) image and recognizes text within them.

    Parameters
    ----------
    dl_image : PIL.Image.Image
        The driver's license image.
    model : YOLO trained model
        The model for object detection.
    processor : TrOCRProcessor
        The processor for text recognition.
    trocr_model : VisionEncoderDecoderModel
        The model for text recognition.

    Returns
    -------
    dict
        A dictionary containing recognized text grouped by labels.
    """
    results = model.predict(source=dl_image, augment=True, verbose=False)
    labels = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number', 'Sex']
    boxes = results[0].boxes
    data = {key: [] for key in labels}
    preds = {int(box.cls):[] for box in boxes}
  
    i = -1
    for bounding_box in boxes:
        i+=1
        if bounding_box.conf[0]<0.6:
            continue
        preds[int(bounding_box.cls)].append((float(bounding_box.conf[0]), i))

        
    for key, value in preds.items():
        idx = max(value, key=lambda x: x[0])[1]
        bounding_box = boxes[idx]        
        x1, y1, x2, y2 = bounding_box.xyxy[0].tolist()
        crop_img = dl_image.crop((x1, y1, x2, y2)).convert("RGB")
        crop_img = crop_img.resize((crop_img.width * 4, crop_img.height * 4))
        pixel_values = processor(images=crop_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=4000)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        data[labels[int(bounding_box.cls)]].append(generated_text)
    return data

def read_img(dl_image, model, processor, trocr_model):
    """Detects labels on a Driver's License (DL) image and recognizes text within them.

    Parameters
    ----------
    dl_image : PIL.Image.Image
        The driver's license image.
    model : YOLO trained model
        The model for object detection.
    processor : TrOCRProcessor
        The processor for text recognition.
    trocr_model : VisionEncoderDecoderModel
        The model for text recognition.

    Returns
    -------
    dict
        A dictionary containing recognized text grouped by labels.
    """
    results = model.predict(source=dl_image, augment=True, verbose=False)
    labels = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number', 'Sex']
    boxes = results[0].boxes
    data = {key: [] for key in labels}
    preds = {int(box.cls): [] for box in boxes}
  
    i = -1
    for bounding_box in boxes:
        i += 1
        if bounding_box.conf[0] < 0.55:
            continue
        preds[int(bounding_box.cls)].append((float(bounding_box.conf[0]), i))

    # Function to process the cropped image using TrOCR
    def process_image(img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(
            pixel_values,
            num_beams=5,  # Beam search to improve text accuracy
            max_new_tokens=200,  # Adjust for the expected text length
            early_stopping=True  # Stops when a reasonable text sequence is detected
        )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    for key, value in preds.items():
        idx = max(value, key=lambda x: x[0])[1]
        bounding_box = boxes[idx]        
        x1, y1, x2, y2 = bounding_box.xyxy[0].tolist()
        crop_img = dl_image.crop((x1, y1, x2, y2)).convert("RGB")
        
        # Optional: Resize the cropped image for better OCR performance
        crop_img = crop_img.resize((crop_img.width * 4, crop_img.height * 4))

        # Check if this is a label like 'Address' that could have multiple lines
        if labels[int(bounding_box.cls)] == 'Address':
            # Split the image into two parts if it's an address
            width, height = crop_img.size
            half_height = height // 2

            # Crop the top part (e.g., street address)
            img_top = crop_img.crop((0, 0, width, half_height))

            # Crop the bottom part (e.g., city and ZIP code)
            img_bottom = crop_img.crop((0, half_height, width, height))

            # Process the top and bottom images separately
            text_top = process_image(img_top)
            text_bottom = process_image(img_bottom)

            # Combine the results
            generated_text = text_top + "\n" + text_bottom
        else:
            # For other labels, process the entire cropped image directly
            generated_text = process_image(crop_img)

        # Append the recognized text to the corresponding label
        data[labels[int(bounding_box.cls)]].append(generated_text)

    return data



def text_extraction(file, class_model, model, processor, trocr_model, rotate_model=None):
    """Extract text from a Driver's License (DL) ID and save it as pd.DataFrame.

    Parameters
    ----------
    file : str
        Path to the DL ID file.
    class_model : torch.nn.Module
        The ResNet model for ID determination.
    rotate_model : torch.nn.Module
        The ResNet model for position determination.
    model : YOLO trained model
        The YOLO model.
    processor : TrOCRProcessor
        The TrOCRProcessor.
    trocr_model : VisionEncoderDecoderModel
        The VisionEncoderDecoderModel.

    Returns
    -------
    pd.DataFrame or str
        If the type of ID is 'id', returns a DataFrame of recognized text by labels.
        If the type is not 'id', returns an empty DataFrame or None.
    """
    if get_type(class_model, file)=='id':
        rotated_image = rotation(rotate_model, file) if rotate_model else Image.open(file)
        data_read = read_img(rotated_image, model, processor, trocr_model)
        max_length = max(len(v) for v in data_read.values())
        data_read['Folder'] = [file.split(os.path.sep)[-2]] * max_length
        data = pd.DataFrame(data_read)
        return data if not data.empty else None
    else:
        return None