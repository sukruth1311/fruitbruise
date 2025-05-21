import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk

# === Load trained model ===
model = load_model("fruit_classifier_model.h5")

# === Detect bruises using HSV color filtering ===
def detect_bruises(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or could not be loaded.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bruise = np.array([10, 50, 50])
    upper_bruise = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_bruise, upper_bruise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output_img = img.copy()
    cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)

    bruise_area = np.sum(mask) / 255
    total_area = img.shape[0] * img.shape[1]
    bruise_percentage = (bruise_area / total_area) * 100

    if bruise_percentage < 5:
        severity = "Minor"
    elif bruise_percentage < 15:
        severity = "Moderate"
    else:
        severity = "Severe"

    return output_img, bruise_percentage, severity, contours

# === Preprocess input image for classification ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read.")
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Predict fruit class using trained model ===
def predict_fruit_class(model, preprocessed_img):
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100

    class_labels = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Tomato']  # modify if needed
    predicted_class = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

    return predicted_class, confidence

# === Health recommendation logic ===
def get_health_recommendations(fruit_type, bruise_percentage, severity):
    recommendations = []

    if severity == "Minor":
        recommendations.append("✅ Safe to eat - minor surface blemishes")
        recommendations.append("✂️ Trim away bruised areas if desired")
    elif severity == "Moderate":
        recommendations.append("⚠️ Edible but best used soon")
        recommendations.append("🔪 Cut away affected portions before eating")
        recommendations.append("🍳 Consider cooking instead of eating raw")
    else:
        recommendations.append("❌ Not recommended for consumption")
        recommendations.append("🚫 Significant microbial risk in bruised areas")
        recommendations.append("🗑️ Consider discarding")

    if "Apple" in fruit_type:
        recommendations.append("ℹ️ Apples brown quickly when bruised due to oxidation")
    elif "Banana" in fruit_type:
        recommendations.append("ℹ️ Bruised bananas are great for baking")

    recommendations.append("💊 Bruising may reduce vitamin C content by 10–30%")

    return recommendations

# === Generate dictionary report ===
def generate_report(fruit_type, confidence, bruise_percentage, severity, recommendations, bruise_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quality = "Good" if severity == "Minor" else "Fair" if severity == "Moderate" else "Poor"

    report = {
        "timestamp": timestamp,
        "fruit_type": fruit_type,
        "confidence": f"{confidence:.2f}%",
        "bruise_percentage": f"{bruise_percentage:.2f}%",
        "severity": severity,
        "bruise_count": bruise_count,
        "recommendations": recommendations,
        "quality": quality
    }

    return report

# === Visual + Console display ===
def display_results(output_img, report):
    text_img = np.ones((500, 600, 3), dtype=np.uint8) * 255
    y_offset = 40

    cv2.putText(text_img, f"Fruit Analysis Report", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y_offset += 40
    cv2.putText(text_img, f"Detected Fruit: {report['fruit_type']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_offset += 30
    cv2.putText(text_img, f"Confidence: {report['confidence']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_offset += 30
    cv2.putText(text_img, f"Bruise Percentage: {report['bruise_percentage']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if report['quality'] != "Good" else (0, 150, 0), 1)
    y_offset += 30
    cv2.putText(text_img, f"Severity: {report['severity']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if report['quality'] != "Good" else (0, 150, 0), 1)
    y_offset += 30
    cv2.putText(text_img, f"Quality: {report['quality']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if report['quality'] != "Good" else (0, 150, 0), 1)
    y_offset += 40
    cv2.putText(text_img, "Recommendations:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_offset += 30

    for rec in report['recommendations']:
        cv2.putText(text_img, rec, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25

    output_img = cv2.resize(output_img, (600, 500))
    combined_img = np.vstack((output_img, text_img))

    cv2.imshow("Fruit Analysis Results", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n=== FRUIT ANALYSIS REPORT ===")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Fruit Type: {report['fruit_type']} ({report['confidence']} confidence)")
    print(f"Bruise Percentage: {report['bruise_percentage']}")
    print(f"Severity: {report['severity']} ({report['bruise_count']} bruised areas)")
    print(f"Quality: {report['quality']}")
    print("\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"- {rec}")

# === Master function to run everything ===
def analyze_fruit(image_path, model):
    try:
        img = preprocess_image(image_path)
        predicted_class, confidence = predict_fruit_class(model, img)
        output_img, bruise_percentage, severity, contours = detect_bruises(image_path)
        recommendations = get_health_recommendations(predicted_class, bruise_percentage, severity)
        report = generate_report(predicted_class, confidence, bruise_percentage, severity, recommendations, len(contours))
        display_results(output_img, report)
        return report
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# === Example usage ===
if __name__ == "__main__":
    image_path = "fruits-360_dataset_100x100/fruits-360/Test/Apple Red 1/4_100.jpg"  # ← Change this path
    analyze_fruit(image_path, model)
