import os
import cv2
import numpy as np
import time
import argparse


def get_largest_rect_contour(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    rect_contour = None
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        if len(approx) == 4:
            rect_contour = approx
            break

    if rect_contour is None:
        raise Exception("No rect found")
    return rect_contour


def get_ordered_rect_points(contour):
    # the order is, bottom-left, bottom-right, top-right, top-left
    rect = np.zeros((4, 2), dtype="float32")
    s = contour.sum(axis=2)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]
    diff = np.diff(contour, axis=2)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]
    return rect


def deskew(src_img, base_rect, skewed_rect):
    M = cv2.getPerspectiveTransform(skewed_rect, base_rect)
    dest_img = cv2.warpPerspective(
        src_img, M, (int(base_rect[2][0]), int(base_rect[2][1])))
    return dest_img


def get_n_largest_components(image, n):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    sorted_areas_indices = np.argsort(-areas)

    component_images = []
    component_image_x = []

    for i in range(1, n + 1):
        largest_component = sorted_areas_indices[i]

        x = stats[largest_component, cv2.CC_STAT_LEFT]
        y = stats[largest_component, cv2.CC_STAT_TOP]
        width = stats[largest_component, cv2.CC_STAT_WIDTH]
        height = stats[largest_component, cv2.CC_STAT_HEIGHT]

        component_image = np.zeros_like(image)
        component_image[labels == largest_component] = 255
        component_image = component_image[y:y+height, x:x+width]
        component_images.append(component_image)
        component_image_x.append(x)

    # Sort the components by x position
    component_images = [x for _, x in sorted(
        zip(component_image_x, component_images))]

    return component_images


def get_templates(dir_path):
    templates = {}
    for file in os.listdir(dir_path):
        if file.endswith(".jpg"):
            template_image = cv2.imread(os.path.join(dir_path, file), 0)
            filename = os.path.splitext(file)[0]
            templates[filename] = template_image
    # key: filename, value, template_image
    return templates


def get_plate_str(images, templates, logs=False):
    str_num = 0
    plate_str = ""
    for parts in images:
        best_match = None
        best_match_value = 0
        for key, value in templates.items():
            if (str_num < 3 and key.isalpha()) or (str_num >= 3 and key.isdigit()):
                result = cv2.matchTemplate(parts, value, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if best_match is None or max_val > best_match_value:
                    best_match = key
                    best_match_value = max_val
                    
        plate_str += best_match
        str_num += 1
        if logs:
            print(f"best match: {best_match}, value: {best_match_value}")
    return plate_str


def main():
    
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str,
                        help="path to image file", default="img_plate/plate1.jpg")
    parser.add_argument("--template_path", type=str,
                        help="path to template folder", default="template")
    parser.add_argument("--show_image", type=bool,
                        help="show image, Value: True or False", default=False)

    img = cv2.imread(parser.parse_args().image_path)
    
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    edge = cv2.Canny(filtered, 30, 200)

    rect_contour = get_largest_rect_contour(edge)

    ordered_rect_points = get_ordered_rect_points(rect_contour)

    plate_size_match = (int(320), int(150))
    base_plate_points = np.array([[0, 0], [plate_size_match[0], 0], [plate_size_match[0], plate_size_match[1]], [
        0, plate_size_match[1]]], dtype="float32")

    plate = deskew(gray, base_plate_points, ordered_rect_points)

    _, plate = cv2.threshold(
        plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plate = cv2.bitwise_not(plate)

    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1)
    plate = cv2.dilate(plate, kernel, iterations=1)

    num_of_digits = 7

    component_images = get_n_largest_components(plate, num_of_digits)

    template_size = (18, 36)

    for i in range(num_of_digits):
        component_images[i] = cv2.resize(component_images[i], template_size)

    templates = get_templates(parser.parse_args().template_path)

    plate_str = get_plate_str(component_images, templates, logs=True)

    print(f"plate number: {plate_str}")
    
    end_time = time.time()
    
    print(f"execution time: {end_time - start_time}")

    if not parser.parse_args().show_image:
        return

    for i in range(num_of_digits):
        cv2.imshow(f"component {i}", component_images[i])

    res_img = img.copy()
    cv2.drawContours(res_img, [rect_contour], -1, (0, 255, 0), 2)
    # draw plate string on result image
    cv2.putText(res_img, plate_str, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for vertex in rect_contour:
        x, y = vertex[0]
        if (width - x) < 100:
            x = width - 100
        if (height - y) < 30:
            y = height - 30
        cv2.putText(res_img, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('plate', plate)
    cv2.imshow('plate_boarder_img', res_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
