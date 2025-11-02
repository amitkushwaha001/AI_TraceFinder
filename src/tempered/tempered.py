import cv2
import numpy as np
import sys

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform some image processing (e.g., create a heatmap)
    heatmap = cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    cv2.imwrite('src/tempered/heatmap.png', heatmap)

    # Segment the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('src/tempered/segmented_result.png', thresh)

    # Draw bounding boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite('src/tempered/result_with_boxes.png', image)

if __name__ == '__main__':
    main(sys.argv[1])
