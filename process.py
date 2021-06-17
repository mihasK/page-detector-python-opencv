import itertools
from sklearn import cluster
from timebudget import timebudget
import math
import cv2
import numpy as np


from icecream import ic

import config
import imutils

_log_counter = 0
def log_image(img, name):
    ic()
    global _log_counter
    _log_counter += 1
    cv2.imwrite(f'output/{_log_counter}-{name}.jpg', img)



def _draw_hough_lines(lines, img, color=None):
    hough_line_output = img.copy()

    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        n = 5000
        x1 = int(x0 + n * (-b))
        y1 = int(y0 + n * (a))
        x2 = int(x0 - n * (-b))
        y2 = int(y0 - n * (a))

        cv2.line(
            hough_line_output, 
            (x1, y1), 
            (x2, y2), 
            color or (0, 0, 255), 
            2
        )
    
    log_image(hough_line_output, 'hough_line', )

    return hough_line_output


@timebudget
def get_hough_lines_p(edges_img, ALLOWED_SKEW = np.pi/6):
        
    tan = math.tan(ALLOWED_SKEW)
#     ic(tan)
    
    y_max = edges_img.shape[0]
    x_max = edges_img.shape[1]
    
    lines_p = cv2.HoughLinesP(edges_img,
                rho=1, 
                theta=np.pi / 360.0, 
                threshold=90,
                minLineLength=min(edges_img.shape)//4,
                maxLineGap=min(edges_img.shape)//7
                    )
    def far_from_center(val, max_val):
        return abs(val - max_val/2) > max_val / 7
    
    def is_angle_good(l):
        x1,y1,x2,y2 = tuple(l[0])
#         print(l)
        dy = abs(y2-y1); dx = abs(x2-x1)
        if dy < tan * dx:  # horizontal case
#             y_md = y2 + 
            if far_from_center(y1, y_max) and far_from_center(y2, y_max):
                return True        
        elif (dx < tan * dy):
            if far_from_center(x1, x_max) and far_from_center(x2, x_max):
                return True
        return False
    
    lines_p = [
        l for l in lines_p 
        if True#is_angle_good(l)
    ]

    if config.OUTPUT_PROCESS: 
        log_image(create_lines_p_image(edges_img, lines_p), 'hough_line_p', )
    
    return lines_p

@timebudget
def get_hough_lines_from_lines_p(lines_p, ):
    def hesse_form_and_length(l):
    
        x1, y1, x2, y2 = l
        
        length = math.sqrt( (y2-y1)**2 + (x2-x1)**2 )
        
        rho = ((x2-x1)*y1 - x1*(y2-y1))  / length
        
        phi = np.pi/2 if  x2==x1 else math.atan( (y2-y1)/(x2-x1) ) 
        theta = (phi + np.pi/2) % np.pi
        
    #     ic(l)
    #     ic(rho, theta)
        return (rho, theta, length)

    lll = [
        hesse_form_and_length(l[0])
        for l in lines_p
    ]

    return np.asarray(lll)

@timebudget
def get_hough_lines_as_clusters(lines_p, ):
    lll = get_hough_lines_from_lines_p(lines_p)

    kmeans = KMeans(
        n_clusters = 4, 
#             init = init_centers,
        init='k-means++',
        max_iter = 100, 
        n_init = 10, 
        random_state = 0,
        
    ).fit(lll[:,:2], sample_weight=lll[:,2])

    return kmeans.cluster_centers_

@timebudget
def get_hough_lines(edges_img, ALLOWED_SKEW = np.pi/6):
    
    lines_p = get_hough_lines_p(edges_img, ALLOWED_SKEW)

    lines_p_img = create_lines_p_image(edges_img, lines_p)
    
    hl_kwargs = dict(
        image=lines_p_img,
        rho=1, 
        theta=np.pi / 360.0, 
        threshold=140,    
    )
    lines = cv2.HoughLines(**hl_kwargs)
    # lines = itertools.chain(
    #     cv2.HoughLines(**hl_kwargs, min_theta=np.pi - ALLOWED_SKEW),  # vertical lines
    #     cv2.HoughLines(**hl_kwargs, min_theta=np.pi/2 - ALLOWED_SKEW, max_theta=np.pi/2 + ALLOWED_SKEW),  # horizontal
    #     cv2.HoughLines(**hl_kwargs, max_theta=ALLOWED_SKEW),  # vertical lines
    # )
    
    
    res = []
    for line in lines:
        rho, theta = line[0]
        
        good = (theta > np.pi - ALLOWED_SKEW) or (theta < ALLOWED_SKEW) or ( np.pi/2 - ALLOWED_SKEW <theta < np.pi/2 + ALLOWED_SKEW)
        # good = True
        if True:
            res.append(line)
    
    return lines

def create_lines_p_image(img, lines_p):
    lines_p_img = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for l in lines_p:
        x1,y1,x2,y2 = tuple(l[0])
        # tan = math.fabs(x1-x2)/(math.fabs(y1-y2) + 1)
        # if tan1<= tan <= tan2:
        #     continue
        cv2.line(
            lines_p_img,
            (x1,y1),(x2,y2),
            255,
            1
            )
    return lines_p_img




from functools import partial

def find_countours(lines, img):
    im_for_cnt = _draw_hough_lines(lines, img=np.zeros(img.shape[:2]), color=1).astype('uint8')

    cnts, hier = cv2.findContours(im_for_cnt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = partial(cv2.arcLength, closed=True), reverse = True)
    # loop over the contours

    # cnts_drawned = []

    for i, c in enumerate(cnts):
        # im = img.copy()
        # cv2.drawContours(im, [c], -1, (255, 0, 0), 2)
        # cnts_drawned.append((i, im))
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c,  0.02*peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper



    # show_mul(cnts_drawned)

    # res = img.copy()
    # cv2.drawContours(res, [screenCnt], -1, (0, 255, 0), 2)

    return screenCnt

from itertools import product
from sklearn.cluster import KMeans


@timebudget
def find_cluster_lines(lines, img):
    data = np.apply_along_axis(
        lambda r: (r[0], (r[1]+np.pi/4) % np.pi), 
        axis=1,
        arr=np.asarray(lines)
    )
    init_centers = np.asarray(list(product(
        [img.shape[1]//4, 3*img.shape[1]//4],
            
        [np.pi/4, 3*np.pi/4]
        )))

    init_centers = np.concatenate(
        (init_centers,np.zeros(shape=(3,2))),
        axis=0
    )


    kmeans = KMeans(
            n_clusters = 7, 
                init = init_centers,
            # init='k-means++',
            max_iter = 100, 
            n_init = 10, 
            random_state = 0
        ).fit(data)

    clusters = np.apply_along_axis(
        lambda r: (r[0], (r[1]-np.pi/4) % np.pi), 
        axis=1,
        arr=np.asarray(kmeans.cluster_centers_)
    )

    return clusters[:4]



@timebudget
def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


@timebudget
def get_angle_between_lines(line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    assert 0<= theta1 <= np.pi
    assert 0<= theta2 <= np.pi

    return abs(theta1 - theta2)

from itertools import combinations


def _draw_intersections(intersections, lines, img):
    intersection_point_output = img.copy()

    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        n = 5000
        x1 = int(x0 + n * (-b))
        y1 = int(y0 + n * (a))
        x2 = int(x0 - n * (-b))
        y2 = int(y0 - n * (a))

        cv2.line(
            intersection_point_output, 
            (x1, y1), 
            (x2, y2), 
            (0, 0, 255), 
            2
        )

    for point in intersections:
        x, y = point[0]

        cv2.circle(
            intersection_point_output,
            (x, y),
            5,
            (255, 255, 127),
            5
        )

    log_image(intersection_point_output, 'intersection_point_output')


@timebudget
def get_intersections(lines, img):
    """Finds the intersections between groups of lines."""
    intersections = []
    group_lines = combinations(range(len(lines)), 2)
    
    K_1 = config.CORNERS_LIM; K_2 = 1 - config.CORNERS_LIM
    
    X_MAX = img.shape[1]
    X_LIM_1 =  K_1 * X_MAX
    X_LIM_2 =  K_2 * X_MAX
    
    Y_MAX = img.shape[0]
    Y_LIM_1 =  K_1 * Y_MAX
    Y_LIM_2 =  K_2 * Y_MAX
        
    x_in_range = lambda x: x <= X_LIM_1 or x>= X_LIM_2
    y_in_range = lambda y: y <= Y_LIM_1 or y>= Y_LIM_2
    
    
    
    ANGLE_L_1 = math.pi/2 - config.ANGLE_BTW_LINES_ALLOWED 
    ANGLE_L_2 = math.pi/2 + config.ANGLE_BTW_LINES_ALLOWED
    for i, j in group_lines:
        line_i, line_j = lines[i], lines[j]
        
        if ANGLE_L_1 < get_angle_between_lines(line_i, line_j) < ANGLE_L_2:
            int_point = intersection(line_i, line_j)
            
            if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
                intersections.append(int_point)

    if config.OUTPUT_PROCESS: 
        _draw_intersections(intersections, lines, img)
    

    return intersections



@timebudget
def draw_green_zone_for_corners(img):
    y_max = img.shape[0]
    x_max = img.shape[1]

    rect_img = np.zeros(shape=img.shape, dtype='uint8')


    rectangles = zip(
        product( (0,x_max), (0, y_max)),
        product( 
            (round(config.CORNERS_LIM*x_max), round(x_max-config.CORNERS_LIM*x_max)), 
            (round(config.CORNERS_LIM*y_max), round(y_max-config.CORNERS_LIM*y_max))
        )
    )
    for r in rectangles:
        ic(r)
        cv2.rectangle(
            rect_img, 
            pt1= r[0],
            pt2=r[1],
            color=(255, 0, 0), 
            thickness=cv2.FILLED,
        )
        img =  cv2.addWeighted(img, 1, rect_img, 0.25, 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = img.shape[0]/800
    thickness = img.shape[0]//500
    cv2.putText(img, 'Put page corners', (30,img.shape[0]//4), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, 'into highlighted zones', (30,3*img.shape[0]//4), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    return img


@timebudget
def find_quadrilaterals(img, intersections, ):
    
    shift_k = 0.2
    
    init_centers = np.array([
        [shift_k*img.shape[0], shift_k * img.shape[1]],
        [(1-shift_k)*img.shape[0], shift_k * img.shape[1]],
        [shift_k*img.shape[0], (1-shift_k) * img.shape[1]],
        [(1-shift_k)*img.shape[0], (1-shift_k) * img.shape[1]],
    ])
    X = np.array([[point[0][0], point[0][1]] for point in intersections])
    kmeans = KMeans(
        n_clusters = 4, 
        init = init_centers,
        max_iter = 100, 
        # n_init = 10, 
        random_state = 0
    ).fit(X)

    points = [center.tolist() for center in kmeans.cluster_centers_]
    return points
    # if config.OUTPUT_PROCESS: self._draw_quadrilaterals(self._lines, kmeans)


def draw_quadrilaterals(img, lines, points):
    grouped_output = img.copy()

    for idx, line in enumerate(lines):
        rho, theta = line
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        n = 5000
        x1 = int(x0 + n * (-b))
        y1 = int(y0 + n * (a))
        x2 = int(x0 - n * (-b))
        y2 = int(y0 - n * (a))

        cv2.line(
            grouped_output, 
            (x1, y1), 
            (x2, y2), 
            (0, 0, 255), 
            2
        )
    
    for point in points:
        x, y = point

        cv2.circle(
            grouped_output,
            (int(x), int(y)),
            5,
            (255, 255, 255),
            5
        )

    log_image(grouped_output, 'grouped')






@timebudget
def add_borders_if_needed(lines, img):
    x_max = img.shape[1]
    y_max = img.shape[0]

    def is_left_border(l):
        rho, theta, _ = l
        return abs(rho) < x_max/2 and not (config.ALLOWED_SKEW < theta < np.pi - config.ALLOWED_SKEW)

    # print('Left:', list(filter(is_left_border, lines)))

    if not list(filter(is_left_border, lines)):
        lines = np.append(lines, np.array([0,0,y_max],)).reshape(lines.shape[0]+1, 3)
        
        
    def is_right_border(l):
        rho, theta, _ = l
        return abs(rho) > x_max/2 and not (config.ALLOWED_SKEW < theta < np.pi - config.ALLOWED_SKEW)

    # print('Right:', list(filter(is_right_border, lines)))
    if not list(filter(is_right_border, lines)):
        lines = np.append(lines, np.array([x_max,0,y_max],)).reshape(lines.shape[0]+1, 3)


    def is_top_border(l):
        rho, theta, _ = l
        return abs(rho) < x_max/2 and (np.pi/2 - config.ALLOWED_SKEW < theta < np.pi/2  + config.ALLOWED_SKEW)

    # print('Top:', list(filter(is_top_border, lines)))
    if not list(filter(is_top_border, lines)):
        lines = np.append(lines, np.array([0,np.pi/2,x_max],)).reshape(lines.shape[0]+1, 3)

    def is_bottom_border(l):
        rho, theta, _ = l
        return abs(rho) > x_max/2 and (np.pi/2 - config.ALLOWED_SKEW < theta < np.pi/2  + config.ALLOWED_SKEW)

    # print('Bottom:', list(filter(is_bottom_border, lines)))
    if not list(filter(is_bottom_border, lines)):
        lines = np.append(lines, np.array([y_max,np.pi/2,x_max],)).reshape(lines.shape[0]+1, 3)
    
    return lines



@timebudget
def resize(image):

    if image.shape[0] <= config.IMAGE_HEIGHT: return image

    ratio = round(config.IMAGE_HEIGHT / image.shape[0], 3)
    width = int(image.shape[1] * ratio)
    dim = (width, config.IMAGE_HEIGHT)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR) 
    if config.OUTPUT_PROCESS: log_image(resized, 'resized')
    return resized


@timebudget
def morphological_close(image, kernel_size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(
        image, 
        cv2.MORPH_CLOSE, 
        kernel,
        iterations = 10
    )
    if config.OUTPUT_PROCESS: log_image(closed, 'closed')

    return closed

@timebudget
def canny(image):
    edges = imutils.auto_canny(image)
    if config.OUTPUT_PROCESS: log_image(edges, 'edges')

    return edges



@timebudget
def order_points(pts):  # -> (tl, tr, br, bl)
    """
    Function for getting the bounding box points in the correct
    order

    Params
    pts     The points in the bounding box. Usually (x, y) coordinates

    Returns
    rect    The ordered set of points: (tl, tr, br, bl)
    """

    if isinstance(pts, list):
        pts = np.array(pts)
    # initialzie a list of coordinates that will be ordered such that 
    # 1st point -> Top left
    # 2nd point -> Top right
    # 3rd point -> Bottom right
    # 4th point -> Bottom left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect