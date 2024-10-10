from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import BadRequest
import re
import joblib
import numpy as np
import os
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import tensorflow as tf
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Diagnomixproject'  # Change this to a secure random key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diagnomixprojecta09.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# User model
class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Field for full name
    email = db.Column(db.String(120), unique=True, nullable=False)  # Unique field for email
    phone_number = db.Column(db.String(15), unique=True, nullable=False)  # Unique field for phone number
    username = db.Column(db.String(100), unique=True, nullable=False)  # Unique field for username
    password_hash = db.Column(db.String(128), nullable=False)  # Field for password hash
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Prediction Result model
class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(50), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))




@app.route('/')
@login_required
def home():
    """Render the home page."""
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')


@app.route('/services')
def services():
    """Render the services page."""
    return render_template('services.html')


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html')


# Disease information routes
@app.route('/heart-disease')
def whd():
    """Render the heart disease information page."""
    return render_template('wheart.html')


@app.route('/liver-disease')
def wld():
    """Render the liver disease information page."""
    return render_template('wliver.html')


@app.route('/kidney-disease')
def wkd():
    """Render the kidney disease information page."""
    return render_template('wkidney.html')


@app.route('/pneumonia')
def wpn():
    """Render the pneumonia information page."""
    return render_template('wpn.html')


@app.route('/breast-cancer')
def wbc():
    """Render the breast cancer information page."""
    return render_template('wbc.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone_number = request.form.get('phonenumber')
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email format')
            return redirect(url_for('register'))

        # Validate phone number (e.g., check if it's numeric and has 10-15 digits)
        if not re.match(r"^\d{10}$", phone_number):
            flash('Invalid phone number. It must be 10 digits long.')
            return redirect(url_for('register'))

        # Validate password constraints
        if not validate_password(password):
            flash(
                'Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        existing_phone = User.query.filter_by(phone_number=phone_number).first()

        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))
        if existing_email:
            flash('Email address already exists')
            return redirect(url_for('register'))
        if existing_phone:
            flash('Phone number already exists')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, phone_number=phone_number, name=name)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


def validate_password(password):
    """Check password complexity requirements."""
    if (len(password) < 8 or
            not re.search(r"[A-Z]", password) or  # At least one uppercase letter
            not re.search(r"[a-z]", password) or  # At least one lowercase letter
            not re.search(r"[0-9]", password) or  # At least one digit
            not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):  # At least one special character
        return False
    return True


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle password reset request."""
    if request.method == 'POST':
        username = request.form.get('username')
        user = User.query.filter_by(username=username).first()

        if user:
            # If the user exists, render the password reset form
            return render_template('reset_password.html', user=user)
        else:
            flash('Username not found.', 'error')

    return render_template('forgot_password.html')




@app.route('/update-password', methods=['POST'])
def update_password():
    """Update the user's password."""
    user_id = request.form.get('user_id')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # Fetch the user from the database
    user = User.query.get(user_id)

    # Check if the user exists
    if not user:
        flash('User not found. Unable to update password.', 'error')
        return render_template('reset_password.html')  # Redirect to reset password page

    # Check if the new password is valid
    if not validate_password(new_password):
        flash('Password must be at least 8 characters long and include at least one uppercase letter, one lowercase letter, one digit, and one special character.', 'error')
        return render_template('reset_password.html', user=user)

    # Check if the new password is the same as the current password
    if user.check_password(new_password):
        flash('New password cannot be the same as the current password.', 'error')
        return render_template('reset_password.html', user=user)

    # Check if the new password matches the confirm password
    if new_password != confirm_password:
        flash('Passwords do not match.', 'error')
        return render_template('reset_password.html', user=user)

    # If all checks pass, update the password
    user.set_password(new_password)  # Hash the new password
    db.session.commit()
    flash('Your password has been updated successfully. You can now log in.', 'success')
    return redirect(url_for('login'))


def check_password(self, password):
    """Check if the provided password matches the stored password."""
    return check_password_hash(self.password_hash, password)

def set_password(self, password):
        """Set the user's password (hash it)."""
        self.password_hash = generate_password_hash(password)


# Load the trained models
MODEL_PATHS = {
    'heart': 'models/heart.pkl',
    'breast_cancer': 'models/breast_cancer.pkl',
    'kidney': 'models/kidney.pkl',
    'liver': 'models/liver.pkl',
    'pneumonia': 'models/pn.h5'
}

models = {}


def load_models():
    for name, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            if name == 'pneumonia':
                models[name] = tf.keras.models.load_model(path)
            else:
                models[name] = joblib.load(path)
            logging.info(f"{name.capitalize()} model loaded successfully from {path}")
        except Exception as e:
            logging.error(f"Error loading {name} model: {str(e)}", exc_info=True)
            models[name] = None


load_models()


def create_pdf(data, prediction, disease):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Diagnomix: Intelligent Health Analysis Platform")
    c.drawString(100, height - 80, f"{disease.capitalize()} Prediction Report")



    # Input data
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 160, "Input Data:")

    c.setFont("Helvetica", 12)
    y = height - 180
    for key, value in data.items():
        c.drawString(120, y, f"{key}: {value}")
        y -= 20

    # Draw the prediction result right after the data
    c.setFont("Helvetica-Bold", 12)
    y -= 20  # Adjust spacing before the prediction result
    c.drawString(120, y, f"Prediction: {prediction}")

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

@app.route('/predict/<disease>', methods=['GET'])
def predict_form(disease):
    if disease not in MODEL_PATHS:
        return "Invalid disease type", 404
    return render_template(f'{disease}.html')


@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        if disease not in models:
            raise ValueError(f"Invalid disease type: {disease}")

        model = models[disease]
        if model is None:
            raise ValueError(f"{disease.capitalize()} model not loaded. Cannot make predictions.")

        # Handle form data for non-pneumonia diseases
        if disease != 'pneumonia':
            form_data = {key: request.form.get(key) for key in request.form if key != 'disease'}
            logging.info(f"Received data for {disease}: {form_data}")

            # Convert form data to floats
            try:
                form_data = {key: float(value) for key, value in form_data.items()}
            except ValueError as ve:
                logging.error(f"Error converting form data to float: {str(ve)}")
                return jsonify({"error": "Invalid data format"}), 400

            data = np.array(list(form_data.values())).reshape(1, -1)

        # Handle file upload for pneumonia
        else:
            if 'image' not in request.files:
                logging.error("No image file provided for pneumonia prediction.")
                return jsonify({"error": "No image file provided"}), 400

            image_file = request.files['image']
            if image_file.filename == '':
                logging.error("Empty image file name for pneumonia prediction.")
                return jsonify({"error": "No image selected"}), 400

            # Process the image
            try:
                img = Image.open(image_file).convert('RGB')  # Ensure image is in RGB mode
                img = img.resize((300, 300))  # Resize to match the input shape
                img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
                data = img_array.reshape(1, 300, 300, 3)  # Reshape for model input
            except Exception as e:
                logging.error(f"Error processing image file: {str(e)}")
                return jsonify({"error": "Invalid image file"}), 400

            form_data = {"image": image_file.filename}  # Log the filename

        logging.debug(f"Data shape for {disease} model: {data.shape}")

        # Make prediction
        prediction = model.predict(data)

        # Define result based on disease type
        if disease == 'heart':
            result = 'Heart disease detected' if prediction[0] == 1 else 'No heart disease detected'
        elif disease == 'breast_cancer':
            result = 'You have cancer' if prediction[0] == 1 else 'You don\'t have cancer'
        elif disease == 'kidney':
            result = 'Chronic Kidney Disease detected' if prediction[0] == 1 else 'No Chronic Kidney Disease detected'
        elif disease == 'liver':
            result = 'Liver disease detected' if prediction[0] == 1 else 'No liver disease detected'
        elif disease == 'pneumonia':
            result = 'Pneumonia detected' if prediction[0][0] > 0.5 else 'No pneumonia detected'

        logging.info(f"{disease.capitalize()} prediction made: {result}")

        # Generate and send PDF
        pdf_buffer = create_pdf(form_data, result, disease)
        logging.debug(f"PDF generated successfully for {disease}. Buffer size: {pdf_buffer.getbuffer().nbytes} bytes")

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'{disease}_report.pdf',
            mimetype='application/pdf'
        )

    except ValueError as ve:
        logging.error(f"Invalid input for {disease}: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Prediction error for {disease}: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


# Error handling
@app.errorhandler(404)
def page_not_found(error):
    """Render 404 error page."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Render 500 error page."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, port=5003)
