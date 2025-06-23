import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load trained model
model = load_model("fruit_classifier_model.h5")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read.")
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_fruit_class(model, preprocessed_img):
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100

    class_labels = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Tomato']
    predicted_class = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

    return predicted_class, confidence


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

    recommendations.append("💊 Bruising may reduce vitamin C content by 10-30%")

    return recommendations


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


def analyze_fruit(image_path, model):
    try:
        img = preprocess_image(image_path)
        predicted_class, confidence = predict_fruit_class(model, img)
        output_img, bruise_percentage, severity, contours = detect_bruises(image_path)
        recommendations = get_health_recommendations(predicted_class, bruise_percentage, severity)
        report = generate_report(predicted_class, confidence, bruise_percentage, severity, recommendations,
                                 len(contours))

        # Save annotated image
        annotated_filename = 'annotated_' + os.path.basename(image_path)
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, output_img)

        return report, annotated_filename
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            report, annotated_filename = analyze_fruit(filepath, model)

            if report and annotated_filename:
                return render_template('results.html',
                                       original=filename,
                                       annotated=annotated_filename,
                                       report=report)
            else:
                return render_template('error.html', message="Error processing image")

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)