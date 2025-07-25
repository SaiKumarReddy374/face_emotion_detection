import cv2
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define Focal Loss function
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)
        y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -y_true * (alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)) - \
               (1 - y_true) * ((1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred))
        return K.mean(loss)
    return loss

# Load the trained model with custom loss function
model = load_model('fer_resnet18_focal.h5', custom_objects={'loss': focal_loss()})

# Emotion labels (FER-2013 dataset)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    """Preprocess the face image for model prediction."""
    face_resized = cv2.resize(face, (48, 48))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype('float32') / 255.0  # Normalize
    face_input = np.expand_dims(face_normalized, axis=0)  # Reshape for model
    return face_input.astype('float32')

def predict_emotion(face):
    """Predict the emotion of the detected face."""
    prediction = model.predict(face)
    predicted_class = np.argmax(prediction)
    return emotion_labels[predicted_class]

def process_webcam():
    """Real-time emotion recognition using webcam."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_input = preprocess_face(face)
            emotion = predict_emotion(face_input)

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition (Webcam)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    """Detects emotions from a single image file."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)  # Ensure correct input shape

        print("Face input shape:", face_input.shape)  # Debugging output

        emotion = predict_emotion(face_input)

        # Draw rectangle & label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition (Image)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Select Mode
print("Choose an option:\n1 - Webcam Mode\n2 - Upload Image")
choice = input("Enter your choice: ")

if choice == "1":
    process_webcam()
elif choice == "2":
    image_path = input("Enter image path: ")
    process_image(image_path)
else:
    print("Invalid choice. Please enter 1 or 2.")
