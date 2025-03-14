import cv2
import numpy as np
import pyttsx3
import time
import pytesseract
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# GPIO Setup
GPIO.setmode(GPIO.BCM)
SWITCH_1 = 17  # Object Detection
SWITCH_2 = 27  # OCR
GPIO.setup(SWITCH_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(SWITCH_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 185)  # Adjust speech rate

# Initialize Camera
picam2 = Picamera2()

# Load YOLO Model
model = YOLO("yolov8n")

# Function to speak alerts
def speak(text):
    engine.say(text)
    engine.runAndWait()

def run_obstacle_detection():
    """Obstacle Detection (Default Mode)"""
    print("Running Obstacle Detection")
    picam2.stop()
    picam2.configure(picam2.create_preview_configuration(main={'format': 'RGB888', 'size': (640, 480)}))
    picam2.start()

    edge_history = []
    history_size = 18
    alert_cooldown = 0
    cooldown_threshold = 40

    def is_plain_surface(plain_ratio):
        return plain_ratio > 0.80  

    while True:
        if GPIO.input(SWITCH_1) == GPIO.LOW and GPIO.input(SWITCH_2) == GPIO.LOW:
            speak("Invalid command")
            time.sleep(1)
            continue
        if GPIO.input(SWITCH_1) == GPIO.LOW:
            return "object_detection"
        if GPIO.input(SWITCH_2) == GPIO.LOW:
            return "ocr"

        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 220)

        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        plain_ratio = 1 - (edge_pixels / total_pixels)

        edge_history.append(plain_ratio)
        if len(edge_history) > history_size:
            edge_history.pop(0)
        avg_plain_ratio = np.mean(edge_history)

        if not is_plain_surface(avg_plain_ratio) or (avg_plain_ratio > 0.993 and alert_cooldown == 0):
            speak("Obstacle detected")
            alert_cooldown = cooldown_threshold
            time.sleep(.5)

        if alert_cooldown > 0:
            alert_cooldown -= 1

def run_object_detection():
    """Object Detection Mode"""
    print("Running Object Detection")
    picam2.stop()
    time.sleep(0.5)
    picam2.preview_configuration.main.size = (1080, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    detected_objects = set()
    start_time = time.time()

    while GPIO.input(SWITCH_1) == GPIO.LOW:
        frame = picam2.capture_array()
        results = model(frame)
        current_objects = {model.names[int(box.cls)] for box in results[0].boxes}
        detected_objects.update(current_objects)

        annotated_frame = results[0].plot()
        cv2.imshow("Camera", annotated_frame)
        cv2.waitKey(1)

        if time.time() - start_time >= 3:
            if detected_objects:
                speech_text = f"Detected objects: {', '.join(detected_objects)}"
                speak(speech_text)
                detected_objects.clear()
            start_time = time.time()

    picam2.stop()
    cv2.destroyAllWindows()
    return "obstacle_detection"

def run_ocr():
    """OCR Text Detection Mode"""
    print("Running OCR")
    picam2.stop()
    time.sleep(0.5)
    config = picam2.create_still_configuration(
        main={"size": (800, 600), "format": "RGB888"},
        controls={"ExposureTime": 50000, "AnalogueGain": 1.0}
    )
    picam2.configure(config)
    picam2.start()

    while GPIO.input(SWITCH_2) == GPIO.LOW:
        speak("Command reading")
        print("Waiting 5 seconds before capturing text...")
        time.sleep(5)  

        frame = picam2.capture_array()
        text = pytesseract.image_to_string(frame).strip()

        if text:
            print("Detected Text:", text)
            engine.setProperty("rate", 155)
            speak(text)
            engine.setProperty("rate", 200)
        else:
            print("No text detected")
            speak("No text detected")

        time.sleep(1)  

    picam2.stop()
    cv2.destroyAllWindows()
    return "obstacle_detection"

# Main Control Loop
current_mode = "obstacle_detection"

while True:
    if current_mode == "obstacle_detection":
        current_mode = run_obstacle_detection()
    elif current_mode == "object_detection":
        current_mode = run_object_detection()
    elif current_mode == "ocr":
        current_mode = run_ocr()
