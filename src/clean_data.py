import os
import cv2
from collections import Counter

def clean_image(image_path, show_steps=False):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    eroded = cv2.erode(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=4)
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=2)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show_steps:
        cv2.imshow("Original", image)
        cv2.imshow("Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return binary

def clean_all_images(input_dir='dataset/origin', output_dir='dataset/clean'):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            cleaned = clean_image(input_path)
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

analyze_characters()
