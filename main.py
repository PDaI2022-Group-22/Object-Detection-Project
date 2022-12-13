from RockPaperScissors import Scissor
from EmotionCheck import FaceCheck
from TasteOfGame import TasteRain
from SnakeGame import Snake
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
#import TasteOfGame
from kivy.uix.screenmanager import ScreenManager, Screen
import PeopleCounting
import PeopleCountingVideo


class MainMenu(Screen):
    def AA(self):
       Scissor()
    def DD(self):
        TasteRain()
    def EE(self):
        FaceCheck()
    def FF(self):
        Snake()



class Manager(ScreenManager):
    main = ObjectProperty(None)
    people_counting = ObjectProperty(None)
    people_counting_video = ObjectProperty(None)


class ObjectDetectionApp(App):
    def build(self):
        return Manager()

ObjectDetectionApp().run()
