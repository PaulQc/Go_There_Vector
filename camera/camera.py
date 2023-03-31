# Code from Miguel Grinberg : https://github.com/miguelgrinberg/flask-video-streaming
# With adaptation for my application
# Paul Grenier, 2023
#
import os
import cv2
import time
from camera import BaseCamera, VectorBaseCamera
import io
import anki_vector

current_robot = ''


class Camera(BaseCamera):
    video_source = -1

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()
        # self.video_source = -1

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(-1)
        if not camera.isOpened():
            time.sleep(2)
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera: 2e try failed too.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()


class VectorCamera(VectorBaseCamera):
    current_robot = None

    def __init__(self, robot):
        VectorCamera.current_robot = robot
        super(VectorCamera, self).__init__()

    def close_tread(self):
        VectorBaseCamera.last_access = VectorBaseCamera.last_access - 1

    @staticmethod
    def vector_frames():
        global current_robot
        while True:
            # read current frame
            img = VectorCamera.current_robot.camera.latest_image.raw_image
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)

            yield img_io.getvalue()
