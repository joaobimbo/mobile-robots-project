import json
import queue
import sys
import cv2
import logging
import zmq
import argparse
from utils import VisionProcessor

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cameraId', help='Id of camera', required=False, default='http://192.168.43.1:8080/video') #'http://192.168.98.126:8080/video'
    parser.add_argument("--host", help="Host for zmq socket", required=False, default="*")
    parser.add_argument("--port", help="Port for zmq socket", required=False, type=int, default=5555)
    parser.add_argument("--debug", help="Enables debug messages", action="store_true", required=False)
    args = parser.parse_args()

    # Set up logging configuration
    debug_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=debug_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Server")

    # Starting 0MQ socket to send data to the clients
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.host}:{args.port}")
    logger.info("0MQ Socket is up and ready to receive subscribers!")

    # Start vision process
    vision = VisionProcessor(args.cameraId)
    vision_pipe = vision.get_pipe()
    vision.start()

    # First get with larger time out to make sure that vision processor has been initialized
    try:
        vision_pipe.get(block=True, timeout=20)
    except queue.Empty as e:
        vision.terminate()
        sys.exit("Vision Processor took too long to respond. ABORTING!")

    while True:
        try:
            frame, face_x, face_size, hand_gesture = vision_pipe.get(block=True, timeout=0.5)
            logger.debug("Received processed that from vision process.")
        except queue.Empty as e:
            logger.error("Vision Processor took too long to respond. Skipping.")
            continue

        logger.debug("Sending processed data to the subscribers.")
        sock.send_string(json.dumps([face_x, face_size, hand_gesture]), encoding="utf-8")

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
