
from flask import Flask, render_template, Response, session, request
from camera import Camera, VectorCamera
from go_there_helper import Messages, VectorStatus
import time
import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps
from tracking import TrackingDot
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = 'the random string for Vector in Quebec city'.encode('utf8')
app.vector_status = VectorStatus()
app.messages = Messages()
app.frame = ''

@app.route('/')
def index():
    """Page d'entré pour l'interaction avec Vector"""
    # Crée/initialise des variables pour cette session
    text = app.messages.message['initial']
    if session.get('required_message') is not None:
        text = app.messages.message[session['required_message']]
    session['required_message'] = 'initial'
    return render_template('index.html', text=text)


def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        app.frame = frame  # Pour rendre le frame disponible à la méthode 'move_to'
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_vector_feed(vector_camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = vector_camera.get_vector_frame()
        yield b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/vector_video_feed')
def vector_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    app.vector_camera = VectorCamera(app.robot)
    return Response(gen_vector_feed(app.vector_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/go_there', methods=['POST'])
def go_there():
    """Vérifie si Vector est disponible. Si oui, alors appel la méthode 'move_to' pour
     effectuer le déplacement requis. Si non disponible, alors active le message 'nogo'
     à envoyer à l'utilisateur"""
    print('\n On entre dans la route')
    request_data = request.json
    x, y = int(request_data['x']), int(request_data['y'])
    print('request_data = ', request_data)
    print('x et y = ', x, y)
    print('x + y = ', x + y)
    if app.vector_status.status == 'available':
        app.robot = anki_vector.Robot(serial='00508611')
        try:
            app.robot.connect()
            vector_connected = True
        except:
            print('Unable to connect to Vector')
            session['required_message'] = 'no_go'
            return 'not_available'
        app.robot.camera.init_camera_feed()
        text = 'Hello Paul, I am ready now'
        app.robot.behavior.drive_off_charger()
        app.robot.behavior.say_text(text)
        app.vector_status.connection = 'open'
    else :
        session['required_message'] = 'no_go'
        return app.vector_status.status
    session['desired_pos'] = (x, y)
    app.vector_status.status = 'not_available'
    return "available"  # Signal à js que Vector était disponible et que le déplacement est ammorcé


@app.route('/go')
def go():
    return render_template('go.html', text=app.messages.message['go'])


@app.route('/move_to', methods=['GET'])
def move_to():
    """Method pour effectuer le déplacement requis pour se rendre à l'endroit demandé"""
    (x, y) = session['desired_pos']
    session['move_status'] = 'undergoing'
    print('\n Vector se rend à x, y = ', x, y)
    #
    buff = np.fromstring(app.frame, np.uint8)
    buff = buff.reshape(1, -1)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    #
    track_vector = TrackingDot()
    if track_vector.define_path(img, (x, y)):
        ## For debugging ##
        # frame_marked = img.copy()
        # cv2.circle(frame_marked, (x, y), 3, (0, 0, 255), -1)
        # for color in track_vector.dot_position.keys():
        #     # dot_pos = (track_vector.dot_position[color][0], track_vector.dot_position[color][1])
        #     cv2.circle(frame_marked, track_vector.dot_position[color], 3, (0, 255, 0), -1)
        # cv2.circle(frame_marked, track_vector.position, 2, (0, 0, 255), -1)
        # #cv2.imshow("Marked", frame_marked)
        # #
        # print(f"Rotation de Vector = {track_vector.rotation:.2f}")
        # print(f"Distance à parcourir = {track_vector.distance:.2f}")
        #
        #cv2.waitKey()
        #
        app.robot.behavior.turn_in_place(degrees(track_vector.rotation), speed=degrees(40))
        app.robot.behavior.drive_straight(distance_mm(track_vector.distance), speed_mmps(30))
        session['move_status'] = 'finished'
        app.vector_status.connection = 'close'
        app.vector_status.status = 'available'
        text = 'I am arrived'
        app.robot.behavior.say_text(text)
        #time.sleep(1)
    print('\nJe ferme le tread de la camera de Vector')
    app.vector_camera.close_tread()
    time.sleep(1)
    print("C'est fait \n")

    app.robot.disconnect()
    del app.robot

    #time.sleep(1)
    app.vector_status.connection = 'close'
    #
    session['move_status'] = 'finished'
    app.vector_status.status = 'available'  # Remet disponible après déconnexion de Vector
    print('Ici on sort de move_to')
    return ""


@app.route('/move_status', methods=['GET', 'POST'])
def move_status():
    """Method pour indiquer que le déplacement est terminé et retourner l'utilisateur
    à la page initiale"""
    #try:
    if (session['move_status'] == 'finished') or (app.vector_status.connection == 'close'):
        session['required_message'] = 'fin'
        if app.vector_status.connection == 'unavailable':
            session['required_message'] = 'no_connection'
            app.vector_status.connection = 'close'
        session['move_status'] = ''  # Reset la variable
        return "finished"
    else:
        return "not_finish"


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
