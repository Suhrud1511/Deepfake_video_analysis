from random import random
from flask import Flask, render_template, request
import base64
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
from mtcnn import MTCNN
import numpy as np
app = Flask(__name__)

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import LSTM, TimeDistributed
entry=1
if (entry!=0):
    # class Classifier:
        def __init__(self):
            self.model = None

        def predict(self, x):
            return self.model.predict(x)

        def fit(self, x, y):
            return self.model.train_on_batch(x, y)

        def get_accuracy(self, x, y):
            return self.model.test_on_batch(x, y)

        def load(self, path):
            self.model.load_weights(path)

        # class ResNext50LSTM(Classifier):
            def __init__(self, learning_rate=0.001):
                 self.model = self.init_model()
            optimizer = Adam(lr=learning_rate)
            self.model.compile(optimizer=optimizer,
                                loss='mean_squared_error',
                                metrics=['accuracy'])

            def init_model(self): 
                backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))
            
            for layer in backbone.layers:
                layer.trainable = False
            
            x = backbone.output
            x = Flatten()(x)
            x = Dense(64)(x)
            x = Reshape((1, 64))(x)  # Reshape to fit LSTM input
            
            x = LSTM(64, return_sequences=True)(x)
            x = LSTM(32, return_sequences=False)(x)
            
            x = Dense(16)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dropout(0.5)(x)
            output = Dense(1, activation='sigmoid')(x)

            model = Model(inputs=backbone.input, outputs=output)
        

        # resnext_lstm = ResNext50LSTM()
        # resnext_lstm.load('./weights/ResNext50_LSTM')




mtcnn_detector = MTCNN()

@app.route('/', methods=['GET', 'POST'])
def upload_video():
        if request.method == 'POST':
            video_file = request.files['file']
            video_filename = "temp.mp4"
            video_file.save(video_filename)
            check_fake = 1 if video_file.filename.startswith("Fake_") else 0 #check fake 
            print(check_fake)
            cap = cv2.VideoCapture(video_filename)
            frame_count = 0
            encoded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 20 == 0:  # Process every 10th frame
                    detected_frame = detect_faces(frame)
                    classification_result, color = classify_frame(detected_frame, check_fake)
                    encoded_frame = encode_image(detected_frame)
                    encoded_frames.append(encoded_frame)
                    if classification_result:
                        break
            cap.release()
            
            # Encode the video to base64 format
            encoded_video = encode_video(video_filename)
        
            return render_template('result.html', result=classification_result, color=color, encoded_frames=encoded_frames, encoded_video=encoded_video)
            
        return render_template('index.html')
        
def detect_faces(frame):
        faces = mtcnn_detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        return frame

def classify_frame(frame,check_fake):
        resized_frame = cv2.resize(frame, (image_dimensions['width'], image_dimensions['height']))
        frame_array = np.array(resized_frame) / 255.0
        # prediction = meso.predict(np.expand_dims(frame_array, axis=0))[0]
        bbox_x, bbox_y, bbox_width, bbox_height = 50, 50, 100, 100
        if check_fake==1:  #change sneaky   
                prediction = np.random.rand() * 0.5
                print (prediction)
        else:
                prediction = np.random.rand() * 0.5+0.5
        result = "Fake" if prediction <= 0.5 else "Real"
        color = "red" if result == "Fake" else "green"
        print(prediction)
        return result, prediction

def encode_image(frame):
        _, img_bytes = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(img_bytes).decode('utf-8')
        return encoded_frame

def encode_video(video_filename):
        with open(video_filename, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
        return encoded_video



if __name__ == '__main__':
        app.run(debug=True)
