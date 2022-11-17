import random

import kivy
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
import cv2
import numpy as np
from kivy.uix.widget import Widget


class PeopleCounting(Widget):
    person_count = ObjectProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.img_path = None
        self.image = self.ids.image

    def file_fire_select(self, *args):
        file_selected = args[1][0]
        print("arguments: ", args[1][0])
        # self.ids.image.source = file_selected
        self.img_path = file_selected
        self.generate_image()
        self.show_people_count()

    def show_people_count(self):
        self.label.text = "There are {} people in the image.".format(str(self.person_count))

    def generate_image(self, *args):
        img = cv2.imread(self.img_path)
        ímg_height = img.shape[0]
        img_width = img.shape[1]
        img_blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

        # set of 90 class labels
        class_labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                        "traffic light",
                        "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                        "horse",
                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                        "eye glasses",
                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                        "baseball bat",
                        "baseball glove",
                        "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork",
                        "knife",
                        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                        "pizza",
                        "donut",
                        "cake", "chair", "sofa", "pottedplant", "bed", "mirror", "diningtable", "window", "desk",
                        "toilet",
                        "door", "tv",
                        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                        "refrigerator",
                        "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # Declare a list of colors as an array
        # green, blue, red, cyan, yellow, purple
        # Split based on ',' and for every split, change type to int
        # convert that to a numpy array to apply color mask to the image numpy array
        class_colors = ["0, 255, 0", "0, 0, 255", "255, 0, 0", "255, 255, 0", "0, 255, 255", "255, 0, 255"]
        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
        class_colors = np.array(class_colors)

        # Loading pretrained model from graph and config files
        # input preprocessed blob into model and pass through the model
        # obtain the detection predictions by the model using forward() method
        maskrcnn = cv2.dnn.readNetFromTensorflow('dataset/frozen_inference_graph.pb', 'dataset/mask_rcnn_config.pbtxt')
        maskrcnn.setInput(img_blob)
        (obj_detections_boxes, obj_detections_masks) = maskrcnn.forward(["detection_out_final", "detection_masks"])
        # returned obj_detections_boxes[0, 0, index, 1] , 1 => will have the prediction class index
        # 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates

        # Loop over the detections
        no_of_detections = obj_detections_boxes.shape[2]

        for index in np.arange(0, no_of_detections):
            prediction_confidence = obj_detections_boxes[0, 0, index, 2]
            # take only predictions with confidence more than --%
            if prediction_confidence > 0.50:
                # get the predicted label
                predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
                predicted_class_label = class_labels[predicted_class_index]

                if class_labels[predicted_class_index] == 'person':
                    # obtain the bounding box co-ordinates for actual image from resized image size
                    bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array(
                        [img_width, ímg_height, img_width, ímg_height])
                    (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")

                    # obtain width and height of the bounding box
                    bounding_box_width = end_x_pt - start_x_pt
                    bounding_box_height = end_y_pt - start_y_pt

                    # obtain the bounding box mask co-ordinates for current detection index
                    object_mask = obj_detections_masks[index, predicted_class_index]

                    # resize mask to bounding box width and bounding box height
                    object_mask = cv2.resize(object_mask, (bounding_box_width, bounding_box_height))

                    # minimum threshold value to convert float based mask array to binary
                    # if true respective values will be true and vice versa
                    object_mask = (object_mask > 0.3)

                    # slice the image array based on bounding box rectangle which is the roi
                    object_region_of_interest = img[start_y_pt:end_y_pt, start_x_pt:end_x_pt]
                    # slice the roi array based on the bounding mask
                    object_region_of_interest = object_region_of_interest[object_mask]

                    # give the mask a random color from class_colors array
                    mask_color = random.choice(class_colors)
                    # add transparent color cover to the region of interest
                    roi_color_transparent_cover = ((0.3 * mask_color) + (0.5 * object_region_of_interest)).astype(
                        "uint8")
                    # place the transparent color cover over the actual image
                    img[start_y_pt:end_y_pt, start_x_pt:end_x_pt][object_mask] = roi_color_transparent_cover

                    # convert the color numpy array to a list and apply to text and box
                    mask_color = [int(c) for c in mask_color]

                    # print the prediction in console
                    predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index],
                                                                 prediction_confidence * 100)
                    print("predicted object {}: {}".format(index + 1, predicted_class_label))

                    self.person_count += 1

                    # draw rectangle and text in the image
                    # cv2.rectangle(img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), mask_color, 2)
                    cv2.putText(img, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                mask_color, 1)
                    # convert image to texture
                    buffer = cv2.flip(img, 0).tobytes()
                    # buffer = bytes(buffer)
                    texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                    self.image.texture = texture

        #cv2.imshow("Detection Output", img)
        print(self.person_count, "people in the image")
        #self.person_count = person_count
        # cv2.waitKey(0)


class PeopleCountingApp(App):
    def build(self):
        # Window.clearcolor = (1,1,1,1)
        return PeopleCounting()


if __name__ == '__main__':
    PeopleCountingApp().run()
    cv2.destroyAllWindows()
