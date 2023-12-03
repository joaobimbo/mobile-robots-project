import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from multiprocessing import Process, Queue
from utils.auxils import translate
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ThymioVision(Process):
    def __init__(self, cam_id: int, data_queue: Queue, *, command_queue: Queue = None, debug: bool = False):
        super().__init__()
        # Setting up logger for the ThymioVision
        self.logger = logging.getLogger('ThymioVision')

        # Queues/pipes
        self.data_queue = data_queue
        self.cmd_queue = command_queue

        # Camera variables
        self.cam_id = cam_id
        self.camera = None

        # Debug variables
        self.debug = debug

    def draw_face_info(self, frame: cv2.typing.MatLike, box: tuple, face_center: tuple, distance_est: float) -> cv2.typing.MatLike:
        # Get box variables
        box_x, box_y, box_w, box_h = box
        face_x, face_y = face_center

        # Draw face rectangle
        pt1 = (box_x, box_y)
        pt2 = (box_x + box_w, box_y + box_h)
        cv2.rectangle(frame, pt1, pt2, (3, 252, 57), 2)

        # Put distance text
        cv2.putText(
            frame,
            f"Distance: {distance_est:.2f} units",
            (box_x, box_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (3, 252, 57),
            2)

        # Put detected coordinates text
        cv2.putText(
            frame,
            f"Horizontal Alignment: {face_x}",
            (box_x, box_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (3, 252, 57),
            2)

        return frame

    def map_coordinates(self, frame: cv2.typing.MatLike, x: int , y: int) -> (int, int):
        frame_h, frame_w, frame_channels = frame.shape

        new_x = translate(x, 0, frame_w, -frame_w / 2, frame_w / 2)
        new_y = translate(y, 0, frame_h, frame_h / 2, -frame_h / 2)

        return new_x, new_y

    def run(self):
        # Defining camera (needs to be defined here to prevent pickling from multiprocessing)
        self.camera = cv2.VideoCapture(self.cam_id)

        # Visual detectors
        face_detector = FaceDetector(minDetectionCon=0.7, modelSelection=1)
        hand_detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.9, minTrackCon=0.5)

        # Local variables for gesture filtering
        last_gesture = None
        gesture_counter = 0
        gesture_threshold = 5  # Adjust this threshold as needed

        #Main visual recognition loop
        while True:
            try:
                # Image reading and visual recognitions
                success, frame = self.camera.read()
                frame, faces = face_detector.findFaces(frame, draw=False)
                hands, frame = hand_detector.findHands(frame, draw=True)

                # Handle face recognition
                if faces:
                    face = faces[0]
                    box_x, box_y, box_w, box_h = face["bbox"]
                    face_x, face_y = self.map_coordinates(frame, face["center"][0], face["center"][1])
                    distance = 500/box_w
                    self.draw_face_info(frame, face["bbox"], (face_x, face_y), distance)
                else:
                    face_x = face_y = distance = None

                # Handle hand recognition
                if hands:
                    hand = hands[0]
                    fingers_up = hand_detector.fingersUp(hand)
                    match fingers_up:
                        case [1, 0, 0, 0, 0]:
                            gesture = "THUMBS_UP"
                        case [1, 1, 1, 1, 1]:
                            gesture = "PALM"
                        case [0, 1, 1, 0, 0]:
                            gesture = "PIECE_SIGN"
                        case _:
                            gesture = None
                else:
                    fingers_up = gesture = None

                # Check if the current gesture is the same as the previous one
                if gesture is not None and gesture == last_gesture:
                    gesture_counter += 1
                    if gesture_counter >= gesture_threshold:
                        # Gesture is stable, set the value
                        stable_gesture = gesture
                else:
                    # Reset the counter if the current gesture is different
                    gesture_counter = 0
                    stable_gesture = None

                # Store the current gesture for the next iteration
                last_gesture = gesture

                # Draw cross line in the center of the frame
                frame_h, frame_w, channels = frame.shape
                cv2.line(frame, (0, int(frame_h / 2)), (frame_w, int(frame_h / 2)), (255, 0, 0), 1)
                cv2.line(frame, (int(frame_w / 2), 0), (int(frame_w / 2), frame_h), (255, 0, 0), 1)

                # Send processed data to the pipe
                self.data_queue.put((frame, face_x, face_y, distance, stable_gesture))
                time.sleep(0.01)
            except Exception as e:
                logging.error(e)
                self.camera.release()

if __name__ == "__main__":
    command_queue = Queue()
    data_queue = Queue()
    vision = ThymioVision(1, data_queue)
    vision.daemon = False
    vision.start()
    state = 0
    while True:
        start_time = time.time()
        if not data_queue.empty():
            frame, x, y, distance, fingers_up = data_queue.get()
            got_time = time.time()
            cv2.imshow('Face Tracking', frame)
            print(x, y, distance, fingers_up)

            print(f"Queue get wait time: {(got_time - start_time):.4f}")
            print(f"Elapsed time: {(time.time() - start_time):.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            vision.terminate()
            break
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            state = abs(state - 1)
            command_queue.put(state)

    cv2.destroyAllWindows()
