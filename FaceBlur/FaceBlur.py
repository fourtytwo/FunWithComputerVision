import sys
import os
from os.path import isfile, join

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from optparse import OptionParser

import cv2
import face_recognition

class FaceBlur(object):

    def __init__(self, input_dir, result_dir, file=None, max_blurring_iterations=5,):
        """
        Constructor of FaceBlur Object
        :param input_dir: directory containing images to process
        :param max_blurring_iterations: maximal iterations for bluring
        :param result_dir: directory of resulting images
        """

        self._max_blurring_iterations = max_blurring_iterations
        if input_dir is None:
            self.file = file
        self.input_dir = input_dir

        if result_dir is None and (input_dir is not None):
            self.result_dir=input_dir+"_blurred"
        else:
            self.result_dir = result_dir

    def blur_face(self,image, face_location):
        """
        Bluring face in face_location on image as long as face is visible, or max_iterations are reached
        :param image: to blur out face
        :param face_location: location of face
        :return: image with blurred out area
        """

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        i = 0
        while (not face_recognition.face_locations(image) == []) and i < self._max_blurring_iterations:
            #blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
            blurred = cv2.blur(blurred, (10, 10))
            i += 1
        return blurred,face_location

    def blur_face_helper(self, args):
        """
        Helper function for pooling
        :param args: arugments
        """
        return self.blur_face(*args)

    def write_image(self, image_name , image):
        if self.input_dir is None :
            cv2.imwrite("blur_{}".format(image_name), image)
        else:
            cv2.imwrite(join(self.result_dir, image_name), image)

    def detect_and_blur_faces(self, image_name):
        """
        Detect faces and blur them in image_name
        :param image_name: name of image to process
        :return: processed image
        """

        if self.input_dir is None :
            image = cv2.imread(image_name)
        else:
            image = cv2.imread(os.path.join(self.input_dir,image_name))

        print(image_name)
        face_locs = face_recognition.face_locations(image)

        list = []
        for face_location in face_locs:

            top, right, bottom, left = face_location
            # face_recognition allowes negative numbers, so here we filter them
            right = max(right, 0)
            right = max(right, 0)
            bottom = max(bottom, 0)
            left = max(left, 0)
            face = image[top:bottom, left:right]
            list.append((face,face_location))

        res_L = []
        with ThreadPoolExecutor() as executor:
            res_L = [executor.submit(self.blur_face_helper, x) for x in list]

            for ft in concurrent.futures.as_completed(res_L):
                i, face_location = ft.result()
                top, right, bottom, left = face_location
                # face_recognition allowes negative numbers, so here we filter them
                right = max(right, 0)
                right = max(right, 0)
                bottom = max(bottom, 0)
                left = max(left, 0)
                image[top:bottom, left:right]=i

        self.write_image(image_name,image)

    def detect_and_blur_helper(self, args):
        """
        Helper function for pooling
        :param args: arugments
        """

        self.detect_and_blur_faces(args)

    def start_processing(self):
        """
        Starts processing
        """

        if self.input_dir is None:
            all_imgs = [self.file]
        else:
            all_imgs = [f for f in os.listdir(self.input_dir) if isfile(join(self.input_dir, f))]

        if self.input_dir is not None:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)

        with Pool() as p:
            list = [(img) for img in all_imgs]
            p.map(self.detect_and_blur_helper, list)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i","--input" , dest="input_dir", help="Directory containing images to be processed")
    parser.add_option("-f", "--file" , dest="file", help="File to process")
    parser.add_option("-o","--output" , dest="output_dir", help="Directory to save results" , default=None)
    parser.add_option("-n", "--maxblur", dest="maxblur", help="Maximal iterations of blur atempt", default=5)
    options,args = parser.parse_args()
    if (options.input_dir is None) and (options.file is None):
        print("Missing argument input directory")
        parser.print_help()
        exit()


    fb = FaceBlur(options.input_dir, options.output_dir, file=options.file, max_blurring_iterations=options.maxblur)
    fb.start_processing()
