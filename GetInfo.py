from kivy.app import App
from kivy.app import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label 
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup


class G(GridLayout):
    pass

class GetInfo(App):
    def build(self):
        return G()

    def show_popup():
        layout = GridLayout(cols = 2, row_force_default=True, row_default_height=30)
  
        popupLabel1 = Label(text = "Width of the measurement base")
        popupLabel2 = Label(text = "Height of the measurement base")
        popupLabel3 = Label(text = "Check if correct IPv4 address")
        popupInput1 = TextInput(hint_text="Width in MM")
        popupInput2 = TextInput(hint_text="Height in MM")
        popupInput3 = TextInput(text= "http://192.168.1.162:8080/video")
        submitButton = Button(text = "Submit")
  
        layout.add_widget(popupLabel1)
        layout.add_widget(popupInput1)
        layout.add_widget(popupLabel2)
        layout.add_widget(popupInput2)       
        layout.add_widget(popupLabel3)
        layout.add_widget(popupInput3)
        layout.add_widget(submitButton)
  
        # Instantiate the modal popup and display
        popup = Popup(title ='Object Measurement',
                      content = layout,
                      size_hint =(None, None), size =(500, 300))  
        popup.open()   
  
        # Attach close button press with popup.dismiss action
        submitButton.bind(on_press = popup.dismiss)

if __name__ == '__main__':
    GetInfo().run()