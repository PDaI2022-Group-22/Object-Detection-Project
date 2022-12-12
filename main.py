import classTest

def main():
    object = classTest.Measurement(200,200,"http://192.168.1.162:8080/video")
    
    object.capture()