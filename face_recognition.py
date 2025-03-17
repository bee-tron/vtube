import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import subprocess
import drawmarks
from flask import Flask, Response
import matplotlib.pyplot as plt

width, height, fps = int(640), int(460), 30

face_base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                      output_face_blendshapes=True,
                                      num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

hands_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hands_options = vision.HandLandmarkerOptions(base_options=hands_base_options,
                                      num_hands=2)
hands_detector = vision.HandLandmarker.create_from_options(hands_options)

pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    output_segmentation_masks=True)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

#process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

def get_frames():
  video_capture = cv2.VideoCapture(0)
  while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if(ret):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.flip(small_frame, 1)
        rgb_img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        frame = np.zeros((height, width, 3), np.uint8)
        #frame=cv2.flip(frame, 1)

        # Process face
        mpImg=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        face_results = face_detector.detect(mpImg)
        hands_results = hands_detector.detect(mpImg)
        #pose_results = pose_detector.detect(mpImg)

        frame=drawmarks.draw_face_landmarks_on_image(frame, face_results)
        frame=drawmarks.draw_hand_landmarks_on_image(frame, hands_results)
        #frame=drawmarks.draw_pose_landmarks_on_image(frame, pose_results)
        
        cv2.imshow('Video', frame)
        cv2.waitKey(1)

        #plot_face_blendshapes_bar_graph(face_results.face_blendshapes[0])

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

  video_capture.release()
  cv2.destroyAllWindows()


app = Flask(__name__)

@app.route('/')
def video():
    return Response(get_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=4000)