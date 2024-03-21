import time
import cv2
import logging
import multiprocessing
from utils.auxils import translate
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector

class VisionProcessor(multiprocessing.Process):
    """
    This class reads from a camera and finds faces and hands through cvzone and mediapipe. It returns through a
    multiprocessing pipe, the processed data from the camera.

    In this project in particular the data is:
    - frame: The opencv frame object with the image captured from the camera, and (optionally) drawing on top
    - face_x: The relative position (to the middle of the frame) of center of the face in the 'horizontal axis'
    - face_distance: The face size is a number in no specific metric, that represent how far is the face from the camera
    - stable_gesture: Filtered (for noise/flickering) hand gesture recognition (THUMBS_UP, PALM or PIECE_SIGN)

    To use this vision processor:
    - get_pipe(): Gets the instance of the output queue/pipe, need to retrieve the outputs
    - start(): Starts the process instance and executes run()
    """
    def __init__(self, cam_id: int, *, daemon: bool = True, draw: bool = True):
        """
        Initializes the visual processor instance.

        :param cam_id: The camera ID used by opencv to get videocapture
        :param daemon: Default is True. When true the processor will also terminate when main exits
        :param draw: Default is True. This tells the processor to draw the detected elements into the frame
        """
        super().__init__(daemon=daemon)

        # Setting up logger for the ThymioVision
        self.logger = logging.getLogger('Vision Processor')

        # Queues/pipes
        self.data_queue = multiprocessing.Queue()

        # Camera variables
        self.cam_id = cam_id
        self.camera = None

        # Drawing configurations
        self.draw = draw

        # Visual filter
        self.filter_threshold = 10  # Number of frames to do filter

    def get_pipe(self) -> multiprocessing.Queue:
        """
        Gets the processor output pipeline.

        :return: The output pipeline.
        """
        return self.data_queue

    def draw_face_info(self, frame: cv2.typing.MatLike, box: tuple, face_center: tuple, distance_est: float) -> cv2.typing.MatLike:
        """
        Draws relevant face information and bounding box around the identified face. Horizontal position and distance
        are drawn into the frame.

        :param frame: Opencv frame object to print in.
        :param box: Tuple containing x, y, width and height information for the bounding box.
        :param face_center: Tuple containing x and y of the center of the face.
        :param distance_est: Distance estimate
        :return: The drawn frame object.
        """
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
        """
        This method maps absolute pixel coordinates, to a plane coordinate system in the middle of the frame.

        :param frame: Opencv frame object to map the coordinates to.
        :param x: Absolute x coordinate to map.
        :param y: Absolute y coordinate to map.
        :return: A tuple containing x and y, relative to the middle of the frame.
        """
        # Get frame sizes
        frame_h, frame_w, frame_channels = frame.shape

        # Map absolute pixel coordinates to center relative ones
        new_x = translate(x, 0, frame_w, -frame_w / 2, frame_w / 2)
        new_y = translate(y, 0, frame_h, frame_h / 2, -frame_h / 2)

        return new_x, new_y

    def run(self):
        """
        This method executes the processor logic.

        **ATTENTION:** If you want to run the processor call start() instead. The start() method allows the execution in
         a separate process, run() will execute the visual processor in main process.
        """
        # Defining camera (needs to be defined here to prevent pickling from multiprocessing)
        self.camera = cv2.VideoCapture(self.cam_id)

        # Visual detectors
        face_detector = FaceDetector(minDetectionCon=0.7, modelSelection=1)
        hand_detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.9, minTrackCon=0.5)

        # Local variables for gesture filtering
        last_gesture = None
        gesture_counter = 0

        self.logger.info("Process is up!")
        while True:
            try:
                # Image reading and error handling
                self.logger.debug(f"Reading frame from the camera (cam_id={self.cam_id}). ")
                success, frame = self.camera.read()
                if not success:
                    self.logger.error("Could not read from the camera. Trying again! (If problem persists check camera configurations)")
                    time.sleep(1)
                    continue

                # Finding visual elements in the frame
                self.logger.debug("Searching for visual elements in the frame.")
                frame, faces = face_detector.findFaces(frame, draw=False)
                hands, frame = hand_detector.findHands(frame, draw=self.draw)

                # Handle face recognition
                if faces:
                    self.logger.debug("Faces found! Processing face data.")
                    face = faces[0]
                    box_x, box_y, box_w, box_h = face["bbox"]
                    face_x, face_y = self.map_coordinates(frame, face["center"][0], face["center"][1])
                    distance = 500/box_w
                    # print(f"distance (500/box_w): {distance}")
                    if self.draw:
                        self.draw_face_info(frame, face["bbox"], (face_x, face_y), distance)
                else:
                    face_x = face_y = distance = None

                # Handle hand recognition
                if hands:
                    self.logger.debug("Hands found! Handling hand data")
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
                    stable_gesture = None
                    if gesture_counter >= self.filter_threshold:
                        # Gesture is stable, set the value
                        stable_gesture = gesture
                        self.logger.debug(f"Filtered '{stable_gesture}' gesture detected!")
                else:
                    # Reset the counter if the current gesture is different
                    gesture_counter = 0
                    stable_gesture = None

                # Store the current gesture for the next iteration
                last_gesture = gesture

                if self.draw:
                    # Draw cross line in the center of the frame
                    frame_h, frame_w, channels = frame.shape
                    cv2.line(frame, (0, int(frame_h / 2)), (frame_w, int(frame_h / 2)), (255, 0, 0), 1)
                    cv2.line(frame, (int(frame_w / 2), 0), (int(frame_w / 2), frame_h), (255, 0, 0), 1)

                # Send processed data to the pipe
                self.logger.debug("Sending processed data to the queue/pipe.")
                self.data_queue.put((frame, face_x, distance, stable_gesture))
                time.sleep(0.005)
            except Exception as e:
                self.logger.error(e)
                self.camera.release()
                break


if __name__ == "__main__":
    command_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()
    vision = VisionProcessor(1)
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
