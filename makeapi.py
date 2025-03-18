# Project Structure:
# captcha_solver/
#   ├── main.py
#   ├── models/
#   │   ├── yolo_model2/best.pt
#   │   └── FINALMODELS/captchasolve.h5
#   ├── utils.py
#   └── requirements.txt

# main.py
import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from ultralytics import YOLO
import joblib
import matplotlib.pyplot as plt

# utils.py
def check_letter_tarakom(pic):
    """
    Check if a letter region meets specific density criteria.
    
    Args:
        pic (numpy.ndarray): Image region to check
    
    Returns:
        bool: True if letter meets density criteria, False otherwise
    """
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - howmanyblack <= 22

def preprocess_letter(letter_crop):
    """
    Preprocess a letter image for model prediction.
    
    Args:
        letter_crop (numpy.ndarray): Cropped letter region
    
    Returns:
        numpy.ndarray: Preprocessed letter image
    """
    resized_letter = cv2.resize(letter_crop, (32, 52))
    resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
    return resized_letter

# Load models and encoders at startup
model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
label_encoder = joblib.load('trained_label_encoder.pkl')
yolo_model = YOLO("yolo_model2/best.pt")

# Create FastAPI app
app = FastAPI(
    title="CAPTCHA Solver API",
    description="An API for solving CAPTCHA images using YOLO and Keras",
    version="1.0.0"
)

@app.post("/solve-captcha/", 
          response_model=dict, 
          summary="Solve a CAPTCHA image",
          description="Upload a CAPTCHA image to extract and recognize its letters")
async def solve_captcha(file: UploadFile = File(...)):
    """
    Endpoint to solve CAPTCHA images.
    
    Args:
        file (UploadFile): Uploaded CAPTCHA image file
    
    Returns:
        dict: Solved CAPTCHA text
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_test = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect letter positions using YOLO
        results_test = yolo_model(image_test)
        detections_test = sorted(results_test[0].boxes, key=lambda box: box.xyxy[0][0])
        
        # Extract and preprocess letters
        letters_test = []
        last_one = []
        howmany = 0
        
        for box in detections_test:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            letter_crop = image_test[y1:y2, x1:x2]
            
            if last_one:
                if int(x1) - last_one[0] > 10:
                    resized_letter = preprocess_letter(letter_crop)
                    
                    # Special handling for the 9th letter (index 8)
                    if howmany == 8:
                        if check_letter_tarakom(letter_crop):
                            letters_test.append(resized_letter)
                    elif howmany <= 7:
                        letters_test.append(resized_letter)
                    
                    last_one = [int(x1), int(y1), int(x2), int(y2)]
                    howmany += 1
            else:
                resized_letter = preprocess_letter(letter_crop)
                
                if check_letter_tarakom(letter_crop):
                    letters_test.append(resized_letter)
                
                last_one = [int(x1), int(y1), int(x2), int(y2)]
                howmany += 1
        
        # Predict letters
        allpredicted = []
        for letter in letters_test:
            sample_image = letter.reshape(1, 52, 32, 1)
            predicted_label = model.predict(sample_image)
            predicted_class = label_encoder.inverse_transform([predicted_label.argmax()])
            allpredicted.append(predicted_class[0])
        
        captcha_result = "".join(allpredicted)
        
        return {"captcha": captcha_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Health check endpoint
@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy"}

# requirements.txt content
"""
fastapi
uvicorn
opencv-python-headless
tensorflow
ultralytics
joblib
python-multipart
numpy
matplotlib
"""

# Run with: uvicorn main:app --reload