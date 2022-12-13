

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
#import TasteOfGame
from kivy.uix.screenmanager import ScreenManager, Screen
import PeopleCounting
import PeopleCountingVideo


class MainMenu(Screen):
    def AA(self):
        exec(open('RockPaperScissors.py').read())
    def DD(self):
        exec(open('TasteOfGame.py').read())
    def EE(self):
        exec(open('EmotionCheck.py').read())
    def FF(self):
        exec(open('SnakeGame.py').read())



class Manager(ScreenManager):
    main = ObjectProperty(None)
    people_counting = ObjectProperty(None)
    people_counting_video = ObjectProperty(None)


class ObjectDetectionApp(App):
    def build(self):
        return Manager()

ObjectDetectionApp().run()
