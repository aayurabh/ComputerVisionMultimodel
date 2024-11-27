#all 3 hand, object, drawing 

import cv2
from PIL import Image, ImageTk
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ultralytics import YOLO
import mediapipe as mp
import tkinter as tk
from tkinter import Button
from threading import Thread

# Global variables
running = False
brush_thickness = 5
eraser_thickness = 50
brush_color = (255, 0, 0)  # Default brush color is red
prev_x, prev_y = None, None  # Previous coordinates for drawing
drawing = False  # To control when drawing is happening
eraser_mode = False  # Flag to indicate whether eraser is active
canvas_history = []  # To store drawing actions for undo/redo
undo_stack = []  # Stack for undo actions
canvas = np.zeros((480, 640, 3), np.uint8)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to control system volume
def set_volume(change):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Get current volume
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(0.0, min(1.0, current_volume + change))
    volume.SetMasterVolumeLevelScalar(new_volume, None)

# Function to detect hand gestures with Mediapipe
def recognize_gesture_with_mediapipe(hand_landmarks, frame):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Check thumb up gesture
    if thumb_tip.y < thumb_ip.y:
        cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        set_volume(0.05)  # Increase volume
    # Check thumb down gesture
    elif thumb_tip.y > thumb_ip.y:
        cv2.putText(frame, "Thumbs Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        set_volume(-0.05)  # Decrease volume

# Function to run hand gesture detection using Mediapipe
def run_hand_gesture_detection():
    global running
    running = True
    cap = cv2.VideoCapture(0)

    while running:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                recognize_gesture_with_mediapipe(hand_landmarks, img)

        cv2.imshow('Hand Gesture Detection', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to run object detection
def run_object_detection():
    global running
    running = True
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO("yolo-Weights/yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
    
    # Custom label for "person"
    custom_label = "Person"

    while running:
        success, img = cap.read()
        results = model(img, stream=True)
        object_counts = {class_name: 0 for class_name in classNames}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Check if the class is "person" and replace it with custom label
                if class_name == "person":
                    display_name = custom_label  # Custom display label
                else:
                    display_name = class_name

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # Display the custom label (or the original class name)
                cv2.putText(img, display_name, org, font, fontScale, color, thickness)

                # Update count in the original "person" category
                object_counts[class_name] += 1

        # Display object counts on screen
        y_offset = 20
        for class_name, count in object_counts.items():
            if count > 0:
                text = f"{class_name}: {count}"
                cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 20

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to stop detection
def stop_detection():
    global running
    running = False

# Function to detect whether the hand is in "drawing mode" (index finger extended)
def is_drawing_mode(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    return index_tip.y < index_pip.y

# Function to redraw the canvas after undo/redo
def redraw_canvas():
    global canvas
    canvas[:] = (0, 0, 0)  # Clear the canvas
    for line in canvas_history:
        for pt1, pt2, color, thickness in line:
            cv2.line(canvas, pt1, pt2, color, thickness)

# Create a function to detect if a color button is clicked
def check_color_selection(x, y):
    global brush_color, eraser_mode, canvas
    if 20 < x < 70 and 20 < y < 70:
        brush_color = (0, 0, 255)  # Red
        eraser_mode = False
    elif 80 < x < 130 and 20 < y < 70:
        brush_color = (0, 255, 0)  # Green
        eraser_mode = False
    elif 140 < x < 190 and 20 < y < 70:
        brush_color = (255, 0, 0)  # Blue
        eraser_mode = False
    elif 200 < x < 250 and 20 < y < 70:
        brush_color = (0, 255, 255)  # Yellow
        eraser_mode = False
    elif 260 < x < 310 and 20 < y < 70:
        brush_color = (255, 255, 255)  # Eraser (white)
        eraser_mode = True
    elif 320 < x < 370 and 20 < y < 70:
        canvas[:] = (0, 0, 0)  # Erase All
        canvas_history.clear()
        undo_stack.clear()  # Clear the history for undo/redo
    elif 380 < x < 430 and 20 < y < 70:
        if canvas_history:
            undo_stack.append(canvas_history.pop())  # Move the last action to the undo stack
            redraw_canvas()  # Redraw the canvas without the last action
    elif 440 < x < 490 and 20 < y < 70:
        if undo_stack:
            canvas_history.append(undo_stack.pop())  # Redo the last undone action
            redraw_canvas()  # Redraw the canvas including the redone action

# Draw color palette on the screen
def draw_color_palette(frame):
    cv2.rectangle(frame, (20, 20), (70, 70), (0, 0, 255), -1)  # Red
    cv2.rectangle(frame, (80, 20), (130, 70), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (140, 20), (190, 70), (255, 0, 0), -1)  # Blue
    cv2.rectangle(frame, (200, 20), (250, 70), (0, 255, 255), -1)  # Yellow
    cv2.rectangle(frame, (260, 20), (310, 70), (255, 255, 255), -1)  # Eraser (white)
    cv2.rectangle(frame, (320, 20), (370, 70), (128, 128, 128), -1)  # Erase All (Gray)
    cv2.putText(frame, 'All', (325, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (380, 20), (430, 70), (0, 0, 0), -1)  # Undo (Black)
    cv2.putText(frame, 'Undo', (385, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (440, 20), (490, 70), (0, 0, 0), -1)  # Redo (Black)
    cv2.putText(frame, 'Redo', (445, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Function to run drawing application
def run_drawing_application():
    global running
    running = True
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while running:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            draw_color_palette(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_tip = int(index_finger_tip.x * frame.shape[1])
                    y_tip = int(index_finger_tip.y * frame.shape[0])

                    if is_drawing_mode(hand_landmarks):
                        if y_tip < 100:
                            check_color_selection(x_tip, y_tip)
                        else:
                            if prev_x is None and prev_y is None:
                                prev_x, prev_y = x_tip, y_tip
                            else:
                                if eraser_mode:
                                    cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (0, 0, 0), eraser_thickness)
                                    canvas_history.append([((prev_x, prev_y), (x_tip, y_tip), (0, 0, 0), eraser_thickness)])
                                else:
                                    cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), brush_color, brush_thickness)
                                    canvas_history.append([((prev_x, prev_y), (x_tip, y_tip), brush_color, brush_thickness)])
                                prev_x, prev_y = x_tip, y_tip
                    else:
                        prev_x, prev_y = None, None

            else:
                prev_x, prev_y = None, None

            combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

            cv2.imshow('Drawing App', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Create the GUI window
root = tk.Tk()
root.title("Computer Vision Multimodel Project")
root.geometry("800x600")

# Set the background image
background_image_path = "C:/Users/saura/OneDrive/Pictures/our project bgm.jpg"  # <-- Adjust your image path here
background_image = Image.open(background_image_path)
background_image = background_image.resize((800, 600), Image.LANCZOS)  # Resize image to fit window
background_photo = ImageTk.PhotoImage(background_image)

# Create a label for the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Title label
title_label = tk.Label(root, text="COMPUTER VISION MULTIMODEL PROJECT", font=("Helvetica", 20, "bold"), bg="#2C3E50", fg="white")
title_label.pack(pady=10)

# Project description label
description_label = tk.Label(root, text="This project includes Hand Gesture Detection, Object Detection and Drawing Application.",
                             font=("Helvetica", 12), bg="#2C3E50", fg="yellow")
description_label.pack(pady=5)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=10)

# Hand gesture detection button
hand_gesture_button = tk.Button(button_frame, text="Hand Gesture Detection", 
                                 command=lambda: Thread(target=run_hand_gesture_detection).start(),
                                 bg="#27AE60", fg="white", font=("Helvetica", 14), padx=10, pady=10)
hand_gesture_button.pack(pady=(5, 0))  # Add padding to the top

# Description for hand gesture detection
hand_gesture_label = tk.Label(button_frame, text="increase sound by thumb up 👍 and decrease by thumb down 👎 ", 
                               font=("Helvetica", 10), bg="#2C3E50", fg="white")
hand_gesture_label.pack(pady=(0, 10))  # Add padding to the bottom

# Object detection button
object_detection_button = tk.Button(button_frame, text="Object Detection", 
                                     command=lambda: Thread(target=run_object_detection).start(),
                                     bg="#2980B9", fg="white", font=("Helvetica", 14), padx=10, pady=10)
object_detection_button.pack(pady=(5, 0))  # Add padding to the top

# Description for object detection
object_detection_label = tk.Label(button_frame, text="Realtime object detection, it will detect anything you show ", 
                                   font=("Helvetica", 10), bg="#2C3E50", fg="white")
object_detection_label.pack(pady=(0, 10))  # Add padding to the bottom

# Add drawing application button
drawing_app_button = tk.Button(button_frame, text="Drawing Application", 
                                command=lambda: Thread(target=run_drawing_application).start(),
                                bg="#F39C12", fg="white", font=("Helvetica", 14), padx=10, pady=10)
drawing_app_button.pack(pady=(5, 0))  # Add padding to the top

# Description for drawing
drawing_app_label = tk.Label(button_frame, text="You can draw by using your index finger 👆  ", 
                              font=("Helvetica", 10), bg="#2C3E50", fg="white")
drawing_app_label.pack(pady=(0, 10))  # Add padding to the bottom

# Add stop button
stop_button = tk.Button(button_frame, text="Stop Detection", 
                        command=stop_detection,
                        bg="#E74C3C", fg="white", font=("Helvetica", 14), padx=10, pady=10)
stop_button.pack(pady=(5, 0))  # Add padding to the top

# Description for stop button
stop_button_label = tk.Label(button_frame, text="Stop current detection.", 
                              font=("Helvetica", 10), bg="#2C3E50", fg="white")
stop_button_label.pack(pady=(0, 10))  # Add padding to the bottom


# Run GUI loop
root.mainloop()