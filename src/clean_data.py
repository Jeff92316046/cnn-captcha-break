import os
import cv2
from collections import Counter

import numpy as np


def clean_image(image: cv2.typing.MatLike, show_steps=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    opened = cv2.morphologyEx(
        blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    )
    eroded = cv2.erode(
        opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    dilated = cv2.dilate(
        eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )
    closed = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)),
        iterations=2,
    )
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)
    opened = cv2.morphologyEx(
        blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw only the contours larger than the threshold
    mask_for_large_objects = np.zeros_like(gray)
    for contour in contours:
        # It's good practice to ensure the contour is not empty
        if len(contour) > 0:
            area = cv2.contourArea(contour)
            if np.isscalar(area):  # Check if it's a scalar value
                if area >= 100:
                    # Draw the contour filled onto the mask
                    cv2.drawContours(
                        mask_for_large_objects, [contour], -1, 255, cv2.FILLED
                    )
    cleaned_binary = cv2.bitwise_and(binary, binary, mask=mask_for_large_objects)
    if show_steps:
        cv2.imshow("Original", image)
        cv2.imshow("Binary", cleaned_binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return binary


def clean_all_images(input_dir="dataset/origin", output_dir="dataset/clean"):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = cv2.imread(input_path)
            cleaned = clean_image(image)
            cv2.imwrite(output_path, cleaned)
            print(f"Saved cleaned image to {output_path}")


def analyze_characters(dataset_path="dataset/clean"):
    filenames = [f for f in os.listdir(dataset_path) if f.endswith(".png")]
    counter = Counter()

    for name in filenames:
        label = os.path.splitext(name)[0]
        counter.update(label)

    print("count:")
    for char, count in sorted(counter.items()):
        print(f"{char}: {count}")

    unique_chars = sorted(counter.keys())
    print("char_set")
    print(f"'{''.join(unique_chars)}'")
    return unique_chars
