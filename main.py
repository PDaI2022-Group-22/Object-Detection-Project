from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
#import TasteOfGame



class GridLayoutExample(GridLayout):
     pass
 


class BoxLayoutExample(BoxLayout):
    def AA(self):
        exec(open('RockPaperScissors.py').read())
    def BB(self):
        exec(open('SnakeGame.py').read())
    def DD(self):
        exec(open('TasteOfGame.py').read())
    def EE(self):
        exec(open('EmotionCheck.py').read())
    pass


class ObjectDetectionApp(App):
    
    pass


ObjectDetectionApp().run()
