import cv2
import numpy as np
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
import random
Builder.load_file("PeopleCounting.kv")


class PeopleCounting(Screen):

    person_count = ObjectProperty(0)
    image_size = ObjectProperty([1, 0.2])

    def __init__(self, **kwargs):
        super(PeopleCounting, self).__init__(**kwargs)
        self.capture = None
        self.img_path = None
        self.image = None
        self.label = None
        self.file_selected = None

    def file_fire_select(self, *args):
        if args[1]:
            self.file_selected = args[1][0]
            print("path: ", args[1][0])
            if self.file_selected.endswith((".jpg", ".png", ".jpeg")):
                self.img_path = self.file_selected
                self.generate_image()
                self.show_people_count()
            else:
                print("File format not supported")

    def show_people_count(self):
        print("show people count")
        self.label.text = "People counted: {} people in the image.".format(str(self.person_count))

    def on_enter(self):
        self.label = self.manager.ids.people_counting.ids.label
        self.image = self.manager.ids.people_counting.ids.image

    def on_leave(self, *args):
        if self.image is not None:
            self.image.texture = None
            self.image.color = 1, 1, 1, 0
        self.label.text = "Choose an image file"
        self.manager.ids.people_counting.ids.filechooser.path = "/"
        self.file_selected = None
        self.image_size = [1, 0.2]

    def generate_image(self, *args):
        self.image_size = [1, 2]
        # Load the image to detect, get width, height
        # resize to match input size, convert to blob(binary large object) to pass into model
        img = cv2.imread(self.img_path)
        img_height = img.shape[0]
        img_width = img.shape[1]

        img_blob = cv2.dnn.blobFromImage(img, 0.003922, (416, 416), swapRB=True, crop=False)
        # recommended by Yolo authors, scale factor is 0.003922=1/255, width, height of blob
        # accepted blob sizes are 320x320, 416x416, 609x609. More size means more accuracy but less speed.

        # set of 80 class labels
        class_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
                        "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
                        "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
                        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                        "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
                        "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                        "cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tv",
                        "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                        "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

        # Declare List of colors as an array Green, Blue, Red, cyan, yellow, purple
        class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255", "255,0,255"]
        # Split based on ',' and for every split, change type to int
        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
        # convert that to a numpy array
        class_colors = np.array(class_colors)

        # Loading pretrained model from weights and config files
        # input preprocessed blob into model and pass through the model
        # obtain the detection predictions by the model using forward() method
        # add a folder dataset into the project root containing .cfg and .weights files
        yolo_model = cv2.dnn.readNetFromDarknet('dataset/yolov3.cfg', 'dataset/yolov3.weights')
        # get layers from the yolo network
        yolo_layer_names = yolo_model.getLayerNames()
        yolo_layer_id = yolo_model.getLayerId(yolo_layer_names[-1])
        print(yolo_layer_names)
        # loop and find the last layer (the output layer) of yolo network
        yolo_output_layer = [yolo_layer_names[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

        # input preprocessed blob into model and pass through the model
        yolo_model.setInput(img_blob)
        # obtain the detection layers by forwarding through till the output layer
        obj_detection_layers = yolo_model.forward(yolo_output_layer)

        # non-max suppression to avoid multiple boxes on a single object
        # declare a list for class_id, box center, width and height and confidences
        class_ids = []
        boxes = []
        confidences = []

        # loop over each of the layer outputs
        for object_detection_layer in obj_detection_layers:
            # loop over the detections
            for object_detection in object_detection_layer:
                # obj_detection[0 to 3] will have the bounding box center x, center y, box width and box height
                # obj_detection[4] will have scores for all objects within bounding box
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]

                # take only predictions with confidence more than ?%
                if prediction_confidence > 0.2:
                    # predicted_class = "person"
                    if predicted_class_id == 0:
                        # obtain the bounding box co-ordinates for actual image from resized image size
                        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                        (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                        start_x_pt = int(box_center_x_pt - (box_width / 2))
                        start_y_pt = int(box_center_y_pt - (box_height / 2))

                        # save class id, start x, y, width and height, confidences in lists for nms
                        # confidence must be passed as float and width and height as integers
                        class_ids.append(predicted_class_id)
                        confidences.append(float(prediction_confidence))
                        boxes.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])


        # Applying NMS will return only the selected max value ids while suppressing the non-maximum overlapping bounding boxes
        # NMS confidence set at 0.5 and max suppression threshold for NMS at 0.4
        final_boxes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(final_boxes)
        self.person_count = 0
        # after NMS loop through the detections and draw bounding boxes + text
        for i in final_boxes:
            box = boxes[i]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]

            # get predicted class id and label
            predicted_class_id = class_ids[i]
            predicted_class_label = class_labels[class_ids[i]]
            prediction_confidence = confidences[i]

            # obtain bounding box end co-ordinates
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            # get a random mask color from the numpy array of colors
            box_color = random.choice(class_colors)

            # convert color numpy array to a list and apply to text and box
            box_color = [int(c) for c in box_color]

            # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            print("predicted object {}".format(predicted_class_label))

            self.person_count += 1

            # draw rectangle and text in the image
            cv2.rectangle(img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
            cv2.putText(img, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color)
            # convert image to texture
            buffer = cv2.flip(img, 0).tobytes()
            # buffer = bytes(buffer)
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.color = 1, 1, 1, 1
            self.image.texture = texture
        print(self.person_count, "people in the image")


# class PeopleCountingApp(App):
#     def build(self):
#         return PeopleCounting()
#
#
# PeopleCountingApp().run()
#
# if __name__ == '__main__':
#     PeopleCountingApp().run()
#     cv2.destroyAllWindows()



