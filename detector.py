import process
import config



def detect_points(img):

    # assert img.shape[0] == 960

    


    image_initial = img
    img = img.copy()
        
        
    # Step 1: Process for edge detection
    for k in (3,7,10):
        img = process.morphological_close(img, kernel_size=k)
        img = process.canny(img)
        if img.astype('bool').sum() / (img.shape[0] * img.shape[1]) < 0.02:
            break
    
    # Step 2: Get hough lines
    lines_p = process.get_hough_lines_p(img,)
    lines = process.add_borders_if_needed(
        lines=process.get_hough_lines_from_lines_p(lines_p),
        img=img
    )[:,:2]

    intersections = process.get_intersections(lines, image_initial)

    # Step 4: Get Quadrilaterals
    points =  process.find_quadrilaterals(img, intersections)

    if config.OUTPUT_PROCESS: 
        process.draw_quadrilaterals(img, lines, points)
    
    return process.order_points(points)

