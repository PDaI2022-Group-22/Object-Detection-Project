
from RockPaperScissors import Scissor
from EmotionCheck import FaceCheck
from TasteOfGame import TasteRain
from SnakeGame import Snake
from ObjectMeasurement import ObjectMeasurement
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
#import TasteOfGame
from kivy.uix.screenmanager import ScreenManager, Screen
import PeopleCounting
import PeopleCountingVideo

url = "http://192.168.1.162:8080/video"


class MainMenu(Screen):
    def AA(self):
       Scissor()
    def DD(self):
        TasteRain()
    def EE(self):
        FaceCheck()
    def FF(self):
        Snake()
    def GG(self):
        ObjectMeasurement(url,210,297)

        

class Manager(ScreenManager):
    main = ObjectProperty(None)
    people_counting = ObjectProperty(None)
    people_counting_video = ObjectProperty(None)


class ObjectDetectionApp(App):
    def build(self):
        return Manager()

ObjectDetectionApp().run()
