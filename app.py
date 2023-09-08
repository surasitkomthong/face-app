from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random

app = Flask(__name__)

# Load the emotion detection model
model = load_model('emotion_detection_model_100epochs.h5')

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), grayscale=True)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    result = model.predict(img)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotion_labels[np.argmax(result)]
    return emotion

@app.route('/')
def index():
    return render_template('index.html')


def randomize_happy_song():
    happy_songs = [
        "https://www.youtube.com/embed/TSsfTJ5OxTo?si=YCZKGPRdG4wuCIkr",
        "https://www.youtube.com/embed/ZLR5ZmkM328?si=9N-Yq6yBkZfuh-Ir",
        # Add more song URLs as needed
    ]
    return random.choice(happy_songs)

def randomize_sad_song():
    sad_songs = [
        "https://open.spotify.com/track/49FYlytm3dAAraYgpoJZux?si=a192f4e519f14104",
        "https://open.spotify.com/track/2ENexcMEMsYk0rVJigVD3i?si=e04171d826e649f6",
        "https://open.spotify.com/track/6QV8q0qVAyHjlnqQ39YIe1?si=222d6b7ecea64f5b",
        "https://open.spotify.com/track/2VOomzT6VavJOGBeySqaMc?si=91106965fedb4a23",
        # Add more song URLs as needed
    ]
    return random.choice(sad_songs)

def randomize_neutral_song():
    neutral_songs = [
        "https://open.spotify.com/track/6wf7Yu7cxBSPrRlWeSeK0Q?si=dc4ca6dc1e9d422b",
        "https://open.spotify.com/track/0u2P5u6lvoDfwTYjAADbn4?si=19f2df771d3142ac",
        "https://open.spotify.com/track/4RVwu0g32PAqgUiJoXsdF8?si=53a99587a5de4039",
        # Add more song URLs as needed
    ]
    return random.choice(neutral_songs)

def randomize_angry_song():
    angry_songs = [
        "https://open.spotify.com/track/4xqrdfXkTW4T0RauPLv3WA?si=ff53bf8908a34ab8",
        "https://open.spotify.com/track/5UXJzLFdBn6u9FJTCnoHrH?si=7a499f54f4414543",
        "https://open.spotify.com/track/6KfoDhO4XUWSbnyKjNp9c4?si=cec1ec8812374c28",
        # Add more song URLs as needed
    ]
    return random.choice(angry_songs)


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' in request.files:
        uploaded_image = request.files['image']
        if uploaded_image.filename != '':
            image_path = 'static/uploaded_image.png'
            uploaded_image.save(image_path)
            emotion = predict_emotion(image_path)
            
            if emotion == "Happy":
                random_song_url = randomize_happy_song()
                return render_template('happy.html', image=image_path, random_song_url=random_song_url)
            if emotion == "Sad":
                random_song_url = randomize_sad_song()
                return render_template('sad.html', image=image_path , random_song_url=random_song_url)
            if emotion == "Neutral":
                random_song_url = randomize_neutral_song()
                return render_template('neutral.html', image=image_path, random_song_url=random_song_url)
            if emotion == "Angry":
                random_song_url = randomize_angry_song()
                return render_template('angry.html', image=image_path, random_song_url=random_song_url)
            else:
                # หากไม่ใช่ "Happy" ให้กลับไปที่หน้าหลัก
                return render_template('index.html', result=emotion, image=image_path)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
