import cv2
import numpy as np
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

Builder.load_file("PeopleCountingVideo.kv")


class PeopleCountingVideo(Screen):
    person_count = ObjectProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.img_path = None
        self.image = None
        self.label = None
        self.file_selected = None

    def file_fire_select(self, *args):
        if args[1]:
            self.file_selected = args[1][0]
            print("path: ", args[1][0])
            if self.file_selected.endswith((".mp4", ".avi")):
                self.img_path = self.file_selected
                self.generate_image()
                self.show_people_count()
            else:
                print("File format not supported")

    def on_enter(self):
        self.label = self.manager.ids.people_counting_video.ids.label
        # self.image = self.manager.ids.people_counting_video.ids.video

    def show_people_count(self):
        self.label.text = "People counted: {} people in the video.".format(str(self.person_count))

    def on_leave(self, *args):
        if self.label is not None:
            self.label.text = "Choose a video file"
        # self.image.texture = None
        self.manager.ids.people_counting_video.ids.filechooser.path = "/"
        self.file_selected = None


    def generate_image(self, *args):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        weights = "checkpoints/yolov4-416"
        iou = 0.45
        score = 0.50
        info = True
        count = True
        dont_show = False
        output = False
        output_format = "XVID"

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = 416

        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        self.capture = cv2.VideoCapture(self.img_path)

        out = None

        # get video ready to save locally if flag is set
        if output:
            # by default VideoCapture returns float instead of int
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*output_format)
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        confirmed_tracks = []
        new_id = None
        # while video is running
        while True:
            re, frame = self.capture.read()
            if re:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num += 1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]
            # print("num_objects: {}".format(num_objects))

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            # allowed_classes = list(class_names.values())

            # custom allowed classes
            allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(bboxes, scores, names, features)]

            # initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima suppression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                if confirmed_tracks.count(track.track_id) == 0:
                    confirmed_tracks.append(track.track_id)
                    new_id = track.track_id
                people_count = len(confirmed_tracks)
                cv2.putText(frame, "New Id: {}, Total Count: {}.".format(new_id, people_count),
                            (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                self.person_count = people_count

                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]

                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)

                # if enable info flag then print details about each track
                # if info:
                #     print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                #         str(track.track_id),
                #         class_name, (
                #             int(bbox[0]),
                #             int(bbox[1]),
                #             int(bbox[2]),
                #             int(bbox[3]))))

            # print("List of tracks: {}".format(confirmed_tracks))
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not dont_show:
                cv2.imshow("Output Video", result)

            # buffer = cv2.flip(frame, 0).tostring()
            # texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            # texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            # self.image.texture = texture

            # if output flag is set, save video file
            if output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()


# class PeopleCountingVideoApp(App):
#     def build(self):
#         return PeopleCountingVideo()
#
#
# if __name__ == '__main__':
#     PeopleCountingVideoApp().run()
#     cv2.destroyAllWindows()



