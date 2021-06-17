import cv2
import numpy as np
from numpy.lib.function_base import trim_zeros
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations, product
from collections import defaultdict
from timebudget import timebudget
from icecream import ic
import process
import config
import traceback

import detector

class PageExtractor:
    def __init__(self, is_extract=True):
        self.is_extract = is_extract

    def __call__(self, image):
        # Step 1: Read image from file

        self.orig_img = image.copy()
        self._image = image

        try:
            # Step 2: Preprocess image
            self._processed = img = process.resize(self._image)

            self._intersections = detector.detect_points(self._processed)
            ic(len(self._intersections))
            
            if len(self._intersections) < 4:
                raise ValueError()    
            # Step 3: Deskew and extract page
            return self.create_result_image()
        except:
            traceback.print_exc()
            if config.DRAW_HINTS:
                return process.draw_green_zone_for_corners(self.orig_img)
            return self.orig_img

    @timebudget
    def highlight_detected_page(self, tl, tr, br, bl):
        base_img = self.orig_img.copy()

        pp = []
        for point in (tl, tr, br, bl):
            x, y = point
            pp.append((int(x),int(y)))

            cv2.circle(
                base_img,
                (x, y),
                5,
                (255, 255, 127),
                5
            )
        cv2.polylines(
            base_img, 
            np.array([pp]), True, (255, 0, 0), 5
        )
        
        if config.OUTPUT_PROCESS: 
            process.log_image(base_img, 'points_on_orig_image')
        
        return base_img

    @timebudget
    def create_result_image(self):
        # obtain a consistent order of the points and unpack them
        # individually
        
        # Resize to fit original img
        ratio = round(self._image.shape[0] / self._processed.shape[0], 3)
        def scale(c):
            return int(c*ratio)
        ic(self._intersections)

        pts = np.array([
            (scale(x), scale(y))
            for x, y in self._intersections
        ])
        # rect = self._order_points(pts)

            
        (tl, tr, br, bl) = pts

        if self.is_extract:
            return self.extact_page(tl, tr, br, bl)
        else:
            return self.highlight_detected_page(tl, tr, br, bl)

    @timebudget
    def extact_page(self, tl, tr, br, bl): 
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],                         # Top left point
            [maxWidth - 1, 0],              # Top right point
            [maxWidth - 1, maxHeight - 1],  # Bottom right point
            [0, maxHeight - 1]],            # Bottom left point
            dtype = "float32"               # Date type
        )

        # compute the perspective transform matrix and then apply it

        pts = np.array([tl, tr, br, bl], dtype = "float32")
        ic(dst)
        ic(pts)
        M = cv2.getPerspectiveTransform(pts.astype('float32'), dst)
        warped = cv2.warpPerspective(self.orig_img, M, (maxWidth, maxHeight))

        if config.OUTPUT_PROCESS: process.log_image(warped, 'deskewed')

        # return the warped image
        return warped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python script to detect and extract documents.")

    parser.add_argument(
        '-i',
        '--input-image',
        help = "Image containing the document",
        required = True,
        dest = 'input_image'
    )

    parser.add_argument(
        '-e',
        '--extract-page',
        help = "To extract or just highlight",
        action="store_true",
        required = False,
        dest = 'is_extract',
    )
    
    args = parser.parse_args()

    timebudget.set_quiet()  # don't show measurements as they happen
    # timebudget.report_at_exit()  # Generate report when the program exits


    page_extractor = PageExtractor(
        is_extract=args.is_extract
    )
    timebudget.report()
    

    with timebudget("Finding a page", quiet=True):
        extracted = page_extractor(cv2.imread(args.input_image))
    print(timebudget.report())

    cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
    cv2.imshow("finalImg", extracted)
    cv2.waitKey(0)
