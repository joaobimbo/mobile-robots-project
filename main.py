import logging
import sys
import time
from utils.robot import ThymioRobot
from utils.vision import ThymioVision
from multiprocessing import Queue
import queue

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    logger = logging.getLogger("__main__")
    robot = ThymioRobot(93.5, refreshing_rate=0.01, debug=False)
    vision_pipe = Queue()
    vision = ThymioVision(1, vision_pipe)
    vision.daemon = True
    vision.start()
    try:
        vision_pipe.get(block=True, timeout=20)
    except queue.Empty as e:
        vision.terminate()
        time.sleep(1)
        vision.close()
        logger.critical(f"Failed to start vision process!")
        sys.exit(-1)

    robot.run_loop(vision_pipe)


