
from kivy.app import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label 
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup

class GetInfo(Widget):

    def __init__(self, **kwargs) -> None:
        super(GetInfo, self).__init__(**kwargs)
        # initializing a layout
        layout = GridLayout(cols = 2, row_force_default=True, row_default_height=30)
        # layout widgets
        popupLabel1 = Label(text = "Width of the measurement base")
        popupLabel2 = Label(text = "Height of the measurement base")
        popupLabel3 = Label(text = "Check if correct IPv4 address")
        self.widthM = TextInput(hint_text="Width in MM", multiline=False)
        self.heightM = TextInput(hint_text="Height in MM", multiline=False)
        self.addressM = TextInput(text="http://192.168.1.162:8080/video", multiline=False)
        submitButton = Button(text = "Submit")
        # adding the widgets to layout
        layout.add_widget(popupLabel1)
        layout.add_widget(self.widthM)
        layout.add_widget(popupLabel2)
        layout.add_widget(self.heightM)       
        layout.add_widget(popupLabel3)
        layout.add_widget(self.addressM)
        layout.add_widget(submitButton)
  
        # Instantiate the modal popup and display
        popup = Popup(title ='Object Measurement',
                      content = layout,
                      size_hint =(None, None), size =(500, 300))
        popup.open()   
  
        # Attach close button press with popup.dismiss and showData function call
        submitButton.bind(on_press = popup.dismiss, on_release=self.showData)

    def showData(self, instance):
        widthM = self.widthM.text
        heightM = self.heightM.text
        addressM = self.addressM.text
        return widthM, heightM, addressM
        #print(f'W: {widthM}\nH: {heightM}\nAddress: {addressM}')

if __name__ == '__main__':
    GetInfo().run()