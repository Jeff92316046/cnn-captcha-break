import os
import cv2
import numpy as np
from glob import glob
import imutils
def extract_characters(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel,iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    erode = cv2.erode(opened, kernel,iterations=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dialte = cv2.dilate(erode, kernel,iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    closed = cv2.morphologyEx(dialte, cv2.MORPH_CLOSE, kernel,iterations=2)
    # binary = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 61, 3)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)
    bounding_boxes = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    
    # 確保有超過一個連通組件 (背景本身是 label 0)
    if num_labels > 1:
        binary_cleaned = np.zeros_like(binary)
        
        # 定義一個最大面積閾值。
        # 這個值需要根據你的驗證碼圖片中文字的典型大小和干擾物的最大大小來設定。
        # 如果你的橫線或者大噪點的面積通常大於所有字元的面積，可以設定為一個較大的值。
        # 經驗值：假設一個字元的面積大約是 50-1000，那麼干擾物的面積可能在 1000+
        max_acceptable_component_area = 1500 # 這個值需要根據你的圖片來調整！
                                             # 建議先觀察你的文字和干擾物的面積大小

        for i in range(1, num_labels): # 從 label 1 開始遍歷 (排除背景)
            component_area = stats[i, cv2.CC_STAT_AREA]
            
            # 如果連通組件的面積大於設定的最大可接受值，就排除它
            if component_area > max_acceptable_component_area:
                # 這裡不添加到 binary_cleaned，相當於排除了它
                pass
            else:
                # 否則，保留這個組件
                binary_cleaned[labels == i] = 255
        binary = binary_cleaned.copy()
    cv2.imshow("Binary After Cleaning (Thresholded Area)", binary) # 檢查清理效果

    # 5. 輪廓檢測與初步篩選
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    min_char_area = 100 # 最小字元面積，過濾小噪點
    max_char_area = 5000 # 最大字元面積，防止把整行當一個字元
    min_aspect_ratio = 0.1 # 最小長寬比 (寬/高)，防止過濾掉細長字元
    max_aspect_ratio = 10.0 # 最大長寬比，防止過濾掉扁平字元

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if h == 0: continue # 避免除以零
        aspect_ratio = w / float(h)

        if min_char_area < area < max_char_area and \
           min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            bounding_boxes.append((x, y, w, h))

    # 6. 合併可能屬於同一字元的邊界框 (例如 'Z' 的上下部分)
    # 這部分是處理 'Z' 被拆分的關鍵，如果預處理做得非常好，這部分甚至可以簡化。
    merged_boxes = []
    if bounding_boxes: # 確保列表不為空
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0]) # 按 x 座標排序

        i = 0
        while i < len(bounding_boxes):
            current_x, current_y, current_w, current_h = bounding_boxes[i]
            
            # 這是合併邏輯的核心，需要根據你的字體和斷裂情況調整
            # 判斷兩個框是否可能屬於同一字元：
            # 1. 水平中心點距離很近 (大部分在同一個字元範圍內)
            # 2. 垂直距離不大，但允許有間隙
            
            # 合併的容忍度 (這些值需要經驗性調整)
            # 垂直距離：例如，兩個框的垂直距離不能超過其中一個框高度的 0.8 倍
            # 水平中心點距離：兩個框的水平中心點不能相差超過一個框寬度的 0.3 倍
            
            # 這裡採用一個更簡潔的合併策略：
            # 檢查後續的框是否在當前框的合理水平範圍內，且垂直距離可接受
            
            merge_candidates = [i]
            for j in range(i + 1, len(bounding_boxes)):
                next_x, next_y, next_w, next_h = bounding_boxes[j]

                # 判斷是否水平重疊或非常接近 (考慮筆畫寬度)
                horizontal_overlap_or_proximity = (max(current_x, next_x) < min(current_x + current_w, next_x + next_w) + 5) # 允許少量重疊或間隙

                # 判斷垂直方向上是否距離不遠，且存在部分重疊或非常接近
                vertical_proximity = abs(current_y - next_y) < (current_h + next_h) * 0.7 # 垂直Y座標差異不能太大

                # 如果滿足這些條件，就認為可能是一個字元的一部分
                if horizontal_overlap_or_proximity and vertical_proximity:
                    merge_candidates.append(j)
                else:
                    # 如果下一個框不符合條件，因為已經排序，後面的框更不可能符合
                    break
            
            # 合併所有候選框
            min_x = min([bounding_boxes[idx][0] for idx in merge_candidates])
            max_x = max([bounding_boxes[idx][0] + bounding_boxes[idx][2] for idx in merge_candidates])
            min_y = min([bounding_boxes[idx][1] for idx in merge_candidates])
            max_y = max([bounding_boxes[idx][1] + bounding_boxes[idx][3] for idx in merge_candidates])

            merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
            i = merge_candidates[-1] + 1 # 移動到下一個未處理的框

    bounding_boxes = merged_boxes
    characters = []
    for x, y, w, h in bounding_boxes:
        region = binary[y:y+h, x:x+w]
        if w > 1.5 * h:
            projected = np.sum(region, axis=0)
            inside = False
            start = 0
            splits = []
            for i, val in enumerate(projected):
                if val > 0 and not inside:
                    start = i
                    inside = True
                elif val == 0 and inside:
                    end = i
                    inside = False
                    if end - start > 2:
                        splits.append((start, end))
            if inside:
                splits.append((start, w))
            for start, end in splits:
                char = region[:, start:end]
                characters.append(cv2.resize(char, (28, 28)))
        else:
            characters.append(cv2.resize(region, (28, 28)))
    cv2.imshow("Binary Image", binary)
    for i in range(len(characters)):
        cv2.imshow(f"Character {i}", characters[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return characters

def process_all_images(input_dir="dataset/origin", output_dir="dataset/splited"):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, "*.png"))

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        label = os.path.splitext(filename)[0]
        chars = extract_characters(img_path)

        if len(chars) != len(label):
            print(f"[Warning] {filename} skipped: {len(chars)} chars found, label is {label}")
            continue

        for i, char_img in enumerate(chars):
            char_label = label[i]
            save_dir = os.path.join(output_dir, char_label)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{filename}_{i}.png")
            cv2.imwrite(save_path, char_img)

        print(f"[OK] {filename} processed.")

if __name__ == "__main__":
    process_all_images()
