# Import Required Libraries
import os
import numpy as np
import cv2
import face_detection
from keras.applications import resnet50
from keras.models import Model
from keras.layers import Dense,Dropout,AveragePooling2D,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tqdm


class Mask_Classifier:
    def __init__(self):
        # Initialize a Face Detector
        # Confidence Threshold can be Adjusted, Greater values would Detect only Clear Faces

        self.detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
        # Define Blurring Kernel Size Ranges, a Random Size would be chosen in the Specified Ranges
        # Greater the Size, Higher is the Blurring Effect (Adjustments can be made according to the needs)

        self.motion_blur_kernel_range = (4, 8)
        self.average_blur_kernel_range = (3, 7)
        self.gaussian_blur_kernel_range = (3, 8)

        # Set Blurring Kernels to Use and their associated Probabilities

        self.Blurring_Kernels = ["none", "motion", "gaussian", "average"]
        self.Probs = [0.75, 0.1, 0.05, 0.1]

        # Set the Hyper-Parameters

        self.alpha = 0.00001
        self.n_epochs = 5
        self.mini_batch_size = 32

        self.MODEL_SAVE_PATH = "Models/ResNet50_Classifier.h5"

        # Set Path to the Dataset
        # Faces would be extracted and placed in the specified Directory after Processing

        self.Dataset_PATH = "Training_Data"
        self.Processed_Dataset_PATH = "Processed_Training_Data"

    # Add Motion Blur to an Image in a Random Direction
    def motion_blur(self,img):

        # Choose a Random Kernel Size
        kernel_size = np.random.randint(self.motion_blur_kernel_range[0], self.motion_blur_kernel_range[1])
        kernel = np.zeros((kernel_size, kernel_size))

        # Random Selection of Direction of Motion Blur
        types = ["vertical", "horizontal", "main_diagonal", "anti_diagonal"]
        choice = np.random.choice(types)

        if choice == "vertical":
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size) / kernel_size

        elif choice == "horizontal":
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

        elif choice == "main_diagonal":

            for i in range(kernel_size):
                kernel[i][i] = 1 / kernel_size

        elif choice == "anti_diagonal":

            for i in range(kernel_size):
                kernel[i][kernel_size - i - 1] = 1 / kernel_size

        # Convolve and Return the Blurred Image
        return cv2.filter2D(img, -1, kernel)

    # Add a Random Blur Effect to an Image with a Random Kernel Size (in the Specified Ranges)

    def get_blurred_picture(self, img):

        # Randomly choose a Blurring Technique
        choice = np.random.choice(self.Blurring_Kernels, p=self.Probs)

        # RGB to BGR for OpenCV
        img = img[:, :, ::-1]

        if choice == "none":

            random_blurred_img = img

        elif choice == "motion":

            random_blurred_img = self.motion_blur(img)

        elif choice == "gaussian":

            kernel_size = np.random.randint(self.gaussian_blur_kernel_range[0], self.gaussian_blur_kernel_range[1])

            if kernel_size % 2 == 0:
                kernel_size -= 1

            random_blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        elif choice == "average":

            kernel_size = np.random.randint(self.average_blur_kernel_range[0], self.average_blur_kernel_range[1])
            random_blurred_img = cv2.blur(img, (kernel_size, kernel_size))

        # PreProcess for ResNet50
        preprocessed = resnet50.preprocess_input(random_blurred_img[:, :, ::-1])

        return preprocessed

    def modelTraining(self):
        # Load Pretrained ResNet50 Model (without Last few Layers)
        # Freeze all the Layers

        base_network = resnet50.ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        for layer in base_network.layers:
            layer.trainaibale = False

        # Define the Face Mask Classifier Model by adding a few Layers on top of the ResNet50 Pretrained Model

        face_classifier_network = base_network.output
        face_classifier_network = AveragePooling2D(pool_size=(7, 7), name="Average_Pool_Final")(face_classifier_network)
        face_classifier_network = Flatten(name="Flatten_Final")(face_classifier_network)
        face_classifier_network = Dense(128, activation="relu", name="Dense_Final")(face_classifier_network)
        face_classifier_network = Dropout(0.5, name="Dropout_Final")(face_classifier_network)
        face_classifier_network = Dense(1, activation="sigmoid", name="Sigmoid_Classifier")(face_classifier_network)

        face_mask_classifier = Model(inputs=base_network.input, outputs=face_classifier_network)

        # Create Empty Directories
        os.mkdir(self.Processed_Dataset_PATH)
        os.mkdir(os.path.join(self.Processed_Dataset_PATH, "with_mask"))
        os.mkdir(os.path.join(self.Processed_Dataset_PATH, "without_mask"))

        # Prepare the Data for Training
        # Extract Faces from the Dataset and Save them in the specified Directory

        # There should be 2 Sub-Directories corresponding to Masked and Non-Masked Faces
        paths = ["with_mask", "without_mask"]

        for path in paths:

            curr_path = os.path.join(self.Dataset_PATH, path)

            # Loop through all Images
            for file_name in tqdm.tqdm(os.listdir(curr_path)):

                try:

                    image = cv2.imread(os.path.join(curr_path, file_name))

                    # Detect Faces, Crop and Save
                    detections = self.detector.detect(image[:, :, ::-1])

                    for j in range(len(detections)):
                        face = image[int(detections[j][1]):int(detections[j][3]),
                               int(detections[j][0]):int(detections[j][2])]

                        cv2.imwrite(os.path.join(self.Processed_Dataset_PATH, path) + "/" + file_name, face)

                except:
                    continue

        # Compile the Model

        opt = Adam(learning_rate=self.alpha, decay=self.alpha / self.n_epochs)
        face_mask_classifier.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

        # Define a ImageDataGenerator for Real-Time Data Augmentation
        # Parameters can be Tuned for controlling the Augmentation

        Data_Generator = ImageDataGenerator(horizontal_flip=True,
                                            brightness_range=[0.5, 1.25],
                                            zoom_range=[0.8, 1],
                                            rotation_range=15,
                                            preprocessing_function=self.get_blurred_picture)

        # Create a Data_Generator Instance

        Train_Data_Generator = Data_Generator.flow_from_directory(self.Processed_Dataset_PATH, target_size=(224, 224),
                                                                  class_mode="binary", batch_size=self.mini_batch_size)

        # Train the Model
        face_mask_classifier.fit(x=Train_Data_Generator,
                            steps_per_epoch=(Train_Data_Generator.n // self.mini_batch_size),
                            epochs=self.n_epochs)

        # Test the Model on a Picture

        FILE_PATH = "testImage.jpg"

        img = cv2.imread(FILE_PATH)
        masked_faces = []
        unmasked_faces = []

        # Detect Faces
        detections = self.detector.detect(img[:, :, ::-1])

        if detections.shape[0] > 0:

            for i in range(detections.shape[0]):

                # Get Co-ordinates
                x1 = int(detections[i][0])
                x2 = int(detections[i][2])
                y1 = int(detections[i][1])
                y2 = int(detections[i][3])

                # Predict Output
                face_arr = cv2.resize(img[y1:y2, x1:x2, ::-1], (224, 224), interpolation=cv2.INTER_NEAREST)
                face_arr = np.expand_dims(face_arr, axis=0)
                face_arr = resnet50.preprocess_input(face_arr)
                match = face_mask_classifier.predict(face_arr)

                if match[0][0] < 0.5:
                    masked_faces.append([x1, y1, x2, y2])
                else:
                    unmasked_faces.append([x1, y1, x2, y2])

        # Put Bounding Box on the Faces (Green:Masked,Red:Not-Masked)
        for f in range(len(masked_faces)):
            a, b, c, d = masked_faces[f]
            cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

        for f in range(len(unmasked_faces)):
            a, b, c, d = unmasked_faces[f]
            cv2.rectangle(img, (a, b), (c, d), (0, 0, 255), 2)

        # Show Results
        cv2.imshow(img)

        # Save the Trained Weights to Disk
        face_mask_classifier.save(self.MODEL_SAVE_PATH)