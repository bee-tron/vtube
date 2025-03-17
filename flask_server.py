# filepath: /home/bjoern/Documents/Python/vtube/flask_server.py
from flask import Flask, Response
import cv2
import time
import os

app = Flask(__name__)

def get_frames():
    while True:
        time.sleep(1/30)
        # Wait for the render to complete
        while not os.path.exists('/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render_done'):
            time.sleep(1/30)

        # Remove the done signal
        os.remove('/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render_done')

        # Read the rendered frame
        frame = cv2.imread('/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render.jpg')
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def video():
    print("connection")
    return Response(get_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)