import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

class FaceRecognizer:
    def __init__(self, model_type):
        self.model_type = model_type

        if model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == "cnn":
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(10, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif model_type == "svm":
            self.model = SVC()
        elif model_type == "gan":
            self.model = None
            # TODO: Implement GAN model
        elif model_type == "gabor":
            self.model = None
            # TODO: Implement Gabor Filter model
        else:
            raise NotImplementedError("Model type {} not supported".format(model_type))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

def extract_features(image, model_type):
    if model_type == "knn" or model_type == "svm":
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to 128x128
        resized = cv2.resize(gray, (128, 128))

        # Flatten image
        flattened = resized.flatten()

        return flattened
    elif model_type == "cnn":
        # Convert image to BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize image to 128x128
        resized = cv2.resize(bgr, (128, 128))

        # Expand dimensions
        expanded = np.expand_dims(resized, axis=0)

        return expanded
    elif model_type == "gan" or model_type == "gabor":
        # TODO: Implement feature extraction for GAN and Gabor Filter models
        pass
    else:
        raise NotImplementedError("Model type {} not supported".format(model_type))

if __name__ == "__main__":
    # Load training data
    X_train = []
    y_train = []
    for i in range(100):
        image = cv2.imread("train_data/{}.jpg".format(i))
        label = i // 10
        X_train.append(extract_features(image, model_type="cnn"))
        y_train.append(label)

    # Train model
    face_recognizer = FaceRecognizer(model_type="cnn")
    face_recognizer.train(X_train, y_train)

    # Load test image
    test_image = cv2.imread("test_image.jpg")

    # Extract features from test image
    test_features = extract_features(test_image, model_type="cnn")

    # Predict label of test image
    prediction = face_recognizer.predict(test_features)

    # Print label
    print("Predicted label:", prediction[0])
