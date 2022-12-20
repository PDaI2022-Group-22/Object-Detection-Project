
# Computer vision & Object detection App collection

This project is a collection of different computer vision and object detection applications merged into a one larger application.
The user interface of the application is done using [Kivy](https://kivy.org/), which is an open-source framework for making cross-platform Python 
applications. The project dependencies are listed in the requirements.txt file. To install them run the command: pip install -r requirements.txt.


## People Counting from Image and Video

### People Counting from Image
Entering the People counting from image screen, the computer file system is opened and the user is asked to
select an image file from device storage. People in the image are detected, marked with bounding boxes and counted.
The resulting image and total number of people is displayed in the screen. OpenCV and Numpy libraries are 
used to process the image and a pre-trained YoloV3 model, trained on the MS COCO dataset, is used for object detection.

### People Counting from Video
In the People counting from video screen the user is asked to select a video file. People are detected from the video
using a pre-trained YoloV4 model and object tracking is added to prevent counting each person again in every frame.
The tracking is done with the DeepSort algorithm, which predicts where the bounding boxes of each object will be from frame to frame 
based on the speed and direction of the object's movement. Each time a new person is detected, they are counted into the total amount of 
people. The output video is displayed in a separate window. 

### Possible Sources of Error In Counting People
Errors in detection and tracking can be caused by such things as poor image quality, overlapping of people in the image and
occlusion of a person from view by other people or objects, which can prevent prediction with sufficient confidence or disrupt the tracking 
and cause the person to be given a new id thus counting the same person more than once.

### Requirements to run People Counting
In order to run the people counting features, download folders dataset and checkpoints from [this link](https://oamk-my.sharepoint.com/:f:/r/personal/t0nihe00_students_oamk_fi/Documents/PeopleCountingFiles?csf=1&web=1&e=prge6N),
and add them to the root folder of the project. The dataset folder contains the YoloV3 weights and configuration file, 
and the checkpoints folder contains a TensorFlow model generated from the YoloV4 weights.

Source for the DeepSort algorithm: (https://github.com/theAIGuysCode/yolov4-deepsort).


## Snake Game
After starting the game from the main menu a new window opens showing the live view of the (laptop) webcam - so, ususally the person starting the game. 
As soon as the person raises the hand and the camera detects it the game starts. The indexfinger becomes the head of the snake 
and whatever direction you move the hand to, the tail of the snake follows your index finger - its head. To score points in the game, 
randomly appearing donuts neads to be "eaten" (collected), which at the same time also increases the length of the snake. 
However, the game is over as soon as you lead the snake into its own tail. It then can be restarted by pressing "R" on the keyboard 
or, if wished, "Q" to quit the game and get back to the main menu. 


## Rock Paper Scissors
After starting the game from the main menu a new window opens showing a user interface with both the live view of the webcam and a field for the computer move.
As soon as the players hand gets recognized by the computer, the game can be started by pressing "S" on the keyboard. 
This will start a counter from 0 to 3, where the move shown on "3" is "logged in" as the final player move. At the same time, the computer randomly chooses a move. 
Both moves are then being compared and the score of the winner gets increased by 1. You can play as many rounds as you want, 
just press "S" on the keyboard to start a new round or, if wished, "Q" to quit the game and get back to the main menu.


## Taste of Game
In this game you try to eat the fruits that drop down from the top of the screen netting you points. All non fruits causes game over and game becomes progressivly harder the more points you have. Game restarts by pressing 'r' and quits by pressing 'q' 

INSERT TEXT HERE
## Emotion Checker
Emotion checker uses deepface library to recognize users most dominant expression or at least one that app recognizes as most likely to be correct emotion. App works better if you are clean shaven and with light illuminating users face. Quitting the app works by pressing 'q'
INSERT TEXT HERE
## Object Measurement

#### Usage:
Current state is more of a **proof-of-concept**

Ability to measure width and height properties of **2D** objects placed on the measurement surface using the IP Webcam application to provide the video stream.

Uses image warping with the dimensions of the measurement base to retain the dimensions regardless of the angle that the measurement base is filmed at. 


#### Requirements:

#### IP Webcam:
https://play.google.com/store/apps/details?id=com.pas.webcam&hl=fi
(Google Play Store version)

#### URL with IP Webcam usability:
This can be received from the IP Webcam application as it connects to your local ip address.
Format: "http://192.168.1.162:8080/video".

Important to have the "/video" at the end of the ip:port.

#### Measurement base:
**Current Measurement base preset is A4 paper with dimensions of
210mm X 297mm (in W x H format)**

Measurement base is a flat surface with the dimensions of the surface known.
Format is Width and Height


