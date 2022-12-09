from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
import GetInfo

class GridLayoutExample(GridLayout):
     pass
 


class BoxLayoutExample(BoxLayout):
    def DD(self):
       exec(open('TasteOfGame.py').read())
    def EE(self):
        exec(open('EmotionCheck.py').read())
    def AA(self):
        exec(open('RockPaperScissors.py').read())
    def FF(self):
        result = GetInfo.GetInfo()
        # can now get the popup setting from object with call.
        #print(result.addressM.text)
        

    pass


class ObjectDetectionApp(App):
    
    pass


ObjectDetectionApp().run()
