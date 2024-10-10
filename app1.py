from flask import Flask, render_template, request, send_file, jsonify
import joblib
import numpy as np
import os
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/heart.pkl'

def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        return None

model = load_model()

def create_pdf(data, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Diagnomix: Intelligent Health Analysis Platform")
    c.drawString(100, height - 80, "Heart Disease Prediction Report")

    # Input data
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 120, "Input Data:")

    c.setFont("Helvetica", 12)
    y = height - 140
    for key, value in data.items():
        c.drawString(120, y, f"{key}: {value}")
        y -= 20

    # Prediction result (printed immediately after input data)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y - 20, "Prediction Result:")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, y - 40, f"{prediction}")

    # Warning message
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 100, "Important Notice:")
    c.setFont("Helvetica", 10)
    c.drawString(100, 80, "This prediction is based on a machine learning model and should not be")
    c.drawString(100, 65, "considered as a definitive medical diagnosis. Please consult with a healthcare")
    c.drawString(100, 50, "professional for accurate medical advice and proper diagnosis.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        # Retrieve and validate form data
        form_data = {key: float(request.form[key]) for key in request.form}
        logging.info(f"Received data: {form_data}")

        # Make prediction
        data = np.array(list(form_data.values())).reshape(1, -1)
        prediction = model.predict(data)
        result = 'Heart disease detected' if prediction[0] == 1 else 'No heart disease detected'
        logging.info(f"Prediction made: {result}")

        # Generate and send PDF
        pdf_buffer = create_pdf(form_data, result)
        logging.debug(f"PDF generated successfully. Buffer size: {pdf_buffer.getbuffer().nbytes} bytes")

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='heart_disease_report.pdf',
            mimetype='application/pdf'
        )

    except ValueError as ve:
        logging.error(f"Invalid input: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)