from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np

class FirstWindow(Screen):
    def __init__(self, sm, **kwargs):
        super(FirstWindow, self).__init__(**kwargs)
        self.sm = sm

        layout = BoxLayout(orientation='vertical')
        open_second_button = Button(text='Open Second Window', on_press=self.open_second_window)
        layout.add_widget(open_second_button)

        self.add_widget(layout)

    def open_second_window(self, instance):
        self.sm.current = 'third'  # Change to 'third' instead of 'second'

class ThirdWindow(Screen):  # Renamed from SecondWindow to ThirdWindow
    def __init__(self, sm, **kwargs):
        super(ThirdWindow, self).__init__(**kwargs)
        self.sm = sm

        layout = BoxLayout(orientation='vertical')

        # Create an Image widget to display the camera feed
        self.image_widget = Image()
        layout.add_widget(self.image_widget)

        go_back_button = Button(text='Go Back to First Window', on_press=self.go_back)
        layout.add_widget(go_back_button)

        self.add_widget(layout)

        # Load pre-trained MobileNetSSD model and its classes
        self.net = cv2.dnn.readNetFromCaffe('mns/deploy.prototxt', 'mns/mobilenet_iter_73000.caffemodel')
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                        "train", "tvmonitor", 'mobile phone']

        # Open a connection to the camera (0 is the default camera)
        self.cap = cv2.VideoCapture(0)

        # Set the desired window size for full-screen display
        self.window_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.window_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Schedule the update method to be called periodically
        Clock.schedule_interval(self.on_update, 1 / 30.0)  # 30 FPS

    def go_back(self, instance):
        Clock.unschedule(self.on_update)  # Stop the update when going back
        self.cap.release()
        self.sm.current = 'first'

    def on_update(self, dt):
        # Read a frame from the camera
        ret, frame = self.cap.read()

        # Resize the frame to fit the window size
        frame = cv2.resize(frame, (self.window_width, self.window_height))

        # Convert the frame to a blob and pass it through the network
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        # Loop over the detections and draw boxes around objects
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by confidence
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])

                # Draw the box and label
                box = detections[0, 0, i, 3:7] * np.array([self.window_width, self.window_height, self.window_width, self.window_height])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{self.classes[class_id]}: {confidence:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the Image widget with the processed frame
        self.image_widget.texture = self.get_texture(frame)

    def get_texture(self, frame):
        # Convert the frame to texture
        buf = cv2.flip(frame, 0).tobytes()  # Use tobytes() instead of tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

class CameraApp(App):
    def build(self):
        sm = ScreenManager()

        first_screen = FirstWindow(name='first', sm=sm)
        third_screen = ThirdWindow(name='third', sm=sm)  # Use ThirdWindow instead of SecondWindow

        sm.add_widget(first_screen)
        sm.add_widget(third_screen)

        return sm

if __name__ == '__main__':
    CameraApp().run()
