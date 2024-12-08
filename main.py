"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
import os
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

# importing module
import logging
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img, output_ = self.lanelines.plot(out_img)
        logger.info(output_)
        print(output_)
        return out_img

    def process_image(self, input_path, output_path = None):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        if output_path is not None:
            mpimg.imsave(output_path, out_img)
        return out_img
        
    def process_frame(self, input_path, output_path = None):
        out_img = self.forward(input_path)
        if output_path is not None:
            mpimg.imsave(output_path, out_img)
        return out_img

    def process_video(self, input_path, output_path):
        # clip = VideoFileClip(input_path)
        # out_clip = clip.fl_image(self.forward)
        # out_clip.write_videofile(output_path, audio=False)

        cap = cv2.VideoCapture(input_path)
        if (cap.isOpened()== False):
            logger.error('Error opening video file')

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                processed_img = self.process_frame(frame)
                cv2.imshow('Processed Frame', processed_img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']

    findLaneLines = FindLaneLines()
    if args['--video']:
        findLaneLines.process_video(input, output)
    else:
        findLaneLines.process_image(input, output)


if __name__ == "__main__":
    main()