# Import Required Libraries
import os
import numpy as np
import cv2
import face_detection
from sklearn.cluster import DBSCAN
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import tqdm


class SocialDistancingAndMaskDetection:
    def __init__(self, inpVidFilePath):
        # Path to the Working Environment

        # If using Google Colab (If on a Local Environment, no path required => set BASE_PATH  = "")
        self.BASE_PATH = "../"

        # Path to Input Video File in the BASE_PATH
        self.FILE_PATH = inpVidFilePath

        # Initialize a Face Detector
        # Confidence Threshold can be Adjusted, Greater values would Detect only Clear Faces

        self.detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

        # Load Pretrained Face Mask Classfier (Keras Model)

        self.mask_classifier = load_model("../Models/ResNet50_Classifier.h5")

        # Set the Safe Distance in Pixel Units (Minimum Distance Expected to be Maintained between People)
        # This Parameter would Affect the Results, Adjust according to the Footage captured by CCTV Camera

        self.threshold_distance = 150  # Try with different Values before Finalizing

        ##################################### Analyze the Video ################################################

        # Load YOLOv3
        self.net = cv2.dnn.readNet(self.BASE_PATH + "Models/" + "yolov3.weights", self.BASE_PATH + "Models/" + "yolov3.cfg")

        # Load COCO Classes
        classes = []
        with open(self.BASE_PATH + "Models/" + "coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def prediction(self):

        # Fetch Video Properties
        cap = cv2.VideoCapture(self.BASE_PATH + self.FILE_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Create Directory for Storing Results (Make sure it doesn't already exists !)
        os.mkdir(self.BASE_PATH + "Output")
        # os.mkdir(self.BASE_PATH + "Output/Extracted_Faces")
        # os.mkdir(self.BASE_PATH + "Output/Extracted_Persons")
        # os.mkdir(self.BASE_PATH + "Output/Frames")

        # Initialize Output Video Stream
        out_stream = cv2.VideoWriter(
            self.BASE_PATH + 'Output/Output.mp4',
            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
            fps,
            (int(width), int(height)))

        print("Processing Frames :")
        for frame in tqdm.tqdm(range(int(n_frames))):

            # Capture Frame-by-Frame
            ret, img = cap.read()

            # Check EOF
            if ret == False:
                break;

            # Get Frame Dimentions
            height, width, channels = img.shape

            # Detect Objects in the Frame with YOLOv3
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []

            # Store Detected Objects with Labels, Bounding_Boxes and their Confidences
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Get Center, Height and Width of the Box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Topleft Co-ordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Initialize empty lists for storing Bounding Boxes of People and their Faces
            persons = []
            masked_faces = []
            unmasked_faces = []

            # Work on Detected Persons in the Frame
            for i in range(len(boxes)):
                if i in indexes:

                    box = np.array(boxes[i])
                    box = np.where(box < 0, 0, box)
                    (x, y, w, h) = box

                    label = str(self.classes[class_ids[i]])

                    if label == 'person':

                        persons.append([x, y, w, h])

                        # # Save Image of Cropped Person (If not required, comment the command below)
                        # cv2.imwrite(self.BASE_PATH + "Output/Extracted_Persons/" + str(frame)
                        #             + "_" + str(len(persons)) + ".jpg",
                        #             img[y:y + h, x:x + w])

                        # Detect Face in the Person
                        person_rgb = img[y:y + h, x:x + w, ::-1]  # Crop & BGR to RGB
                        detections = self.detector.detect(person_rgb)

                        # If a Face is Detected
                        if detections.shape[0] > 0:

                            detection = np.array(detections[0])
                            detection = np.where(detection < 0, 0, detection)

                            # Calculating Co-ordinates of the Detected Face
                            x1 = x + int(detection[0])
                            x2 = x + int(detection[2])
                            y1 = y + int(detection[1])
                            y2 = y + int(detection[3])

                            try:

                                # Crop & BGR to RGB
                                face_rgb = img[y1:y2, x1:x2, ::-1]

                                # Preprocess the Image
                                face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                                face_arr = np.expand_dims(face_arr, axis=0)
                                face_arr = preprocess_input(face_arr)

                                # Predict if the Face is Masked or Not
                                score = self.mask_classifier.predict(face_arr)

                                # Determine and store Results
                                if score[0][0] < 0.5:
                                    masked_faces.append([x1, y1, x2, y2])
                                else:
                                    unmasked_faces.append([x1, y1, x2, y2])

                                # # Save Image of Cropped Face (If not required, comment the command below)
                                # cv2.imwrite(self.BASE_PATH + "Output/Extracted_Faces/" + str(frame)
                                #             + "_" + str(len(persons)) + ".jpg",
                                #             img[y1:y2, x1:x2])

                            except:
                                continue

            # Calculate Coordinates of People Detected and find Clusters using DBSCAN
            person_coordinates = []

            for p in range(len(persons)):
                person_coordinates.append(
                    (persons[p][0] + int(persons[p][2] / 2), persons[p][1] + int(persons[p][3] / 2)))

            clustering = DBSCAN(eps=self.threshold_distance, min_samples=2).fit(person_coordinates)
            isSafe = clustering.labels_

            # Count
            person_count = len(persons)
            masked_face_count = len(masked_faces)
            unmasked_face_count = len(unmasked_faces)
            safe_count = np.sum((isSafe == -1) * 1)
            unsafe_count = person_count - safe_count

            # Show Clusters using Red Lines
            arg_sorted = np.argsort(isSafe)

            for i in range(1, person_count):

                if isSafe[arg_sorted[i]] != -1 and isSafe[arg_sorted[i]] == isSafe[arg_sorted[i - 1]]:
                    cv2.line(img, person_coordinates[arg_sorted[i]], person_coordinates[arg_sorted[i - 1]], (0, 0, 255),
                             2)

            # Put Bounding Boxes on People in the Frame
            for p in range(person_count):

                a, b, c, d = persons[p]

                # Green if Safe, Red if UnSafe
                if isSafe[p] == -1:
                    cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (a, b), (a + c, b + d), (0, 0, 255), 2)

            # Put Bounding Boxes on Faces in the Frame
            # Green if Safe, Red if UnSafe
            for f in range(masked_face_count):
                a, b, c, d = masked_faces[f]
                cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

            for f in range(unmasked_face_count):
                a, b, c, d = unmasked_faces[f]
                cv2.rectangle(img, (a, b), (c, d), (0, 0, 255), 2)

            # Show Monitoring Status in a Black Box at the Top
            cv2.rectangle(img, (0, 0), (width, 50), (0, 0, 0), -1)
            cv2.rectangle(img, (1, 1), (width - 1, 50), (255, 255, 255), 2)

            xpos = 15

            string = "Total People = " + str(person_count)
            cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

            string = " ( " + str(safe_count) + " Safe "
            cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

            string = str(unsafe_count) + " Unsafe ) "
            cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            xpos += cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

            string = "( " + str(masked_face_count) + " Masked " + str(unmasked_face_count) + " Unmasked " + \
                     str(person_count - masked_face_count - unmasked_face_count) + " Unknown )"
            cv2.putText(img, string, (xpos, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Write Frame to the Output File
            out_stream.write(img)

            # # Save the Frame in frame_no.png format (If not required, comment the command below)
            # cv2.imwrite(self.BASE_PATH + "Output/Frames/" + str(frame) + ".jpg", img)

            # Use if you want to see Results Frame by Frame
            # cv2_imshow('results',img)

            # Exit on Pressing Q Key
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release Streams
        out_stream.release()
        cap.release()
        cv2.destroyAllWindows()

        # Good to Go!
        print("Done !")


inputVideoFile = "Input/test_video.mp4"
opFilePath = "/Output"
obj = SocialDistancingAndMaskDetection(inputVideoFile)
obj.prediction()