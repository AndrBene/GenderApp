# Face Recognition & Gender Classification Flask App

This project is a web application for face recognition and gender classification using deep learning. Users can upload an image, and the app will detect faces, classify gender (male/female), and display results with bounding boxes and prediction scores.

## Features
- Upload images via a web interface
- Detect faces in uploaded images
- Classify gender for each detected face
- Display results with bounding boxes and prediction scores
- View grayscale and eigenface images for each detection

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AndrBene/GenderApp.git
   cd GenderApp
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   export FLASK_APP=run.py
   flask run
   # or simply
   python main.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage
1. Open your browser and go to `http://127.0.0.1:5000/genderapp`.
2. Upload an image (JPG, JPEG, PNG) with one or more faces.
3. The app will display the image with detected faces, gender predictions, and scores.
4. A report table will show grayscale and eigenface images for each detection.

## Project Structure
- `app/` - Flask app code
- `model/` - Pre-trained models and related files for face recognition and gender classification
- `templates/` - HTML templates
- `static/` - Static files (uploads, predictions, CSS, etc.)
- `requirements.txt` - Python dependencies
- `run.py` - App entry point

## Notes
- Make sure the `static/upload` and `static/predict` folders exist and are writable.
- The face recognition and gender classification models are loaded in `app/face_recognition.py`.

## License
MIT License


