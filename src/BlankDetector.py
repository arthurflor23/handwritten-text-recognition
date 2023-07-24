import cv2
import math
import statistics
import numpy as np
import xgboost as xgb


class BlankDetector:

    def __init__(self, blank_model_weights, WRITE_BAD_TO_OWN_FILE=False, bad_path=None):
        self.blank_model = xgb.XGBClassifier()
        self.blank_model.load_model(blank_model_weights)
        self.WRITE_BAD_TO_OWN_FILE = WRITE_BAD_TO_OWN_FILE
        self.badPath = bad_path

    def predictBlank(self, img):
        # first check if image is a blank snippet
        vertical_crop = int(img.shape[0] * 0.1375)
        horizontal_crop = int(img.shape[1] * 0.2375)

        cropped_image = img[vertical_crop:-vertical_crop, horizontal_crop:-horizontal_crop]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        white_pixels = cv2.countNonZero(closing)
        total_pixels = closing.shape[0] * closing.shape[1]
        white_percent = (white_pixels / total_pixels) * 100
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_xy = [int(cropped_image.shape[1] / 2), int(cropped_image.shape[0] / 2)]

        cnt_dist_from_center = []
        cnt_area = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_dist_from_center.append(math.dist(center_xy, [x, y]) / cropped_image.shape[1])
            cnt_area.append(w * h)

        cnt_avg_size = math.fsum(cnt_area) / len(cnt_area) if len(contours) > 0 else 0
        largest_cnt = max(cnt_area)
        smallest_cnt = min(cnt_area)
        median_cnt = statistics.median(cnt_area)

        cnt_avg_dist = math.fsum(cnt_dist_from_center) / len(cnt_dist_from_center) if len(contours) > 0 else 0
        largest_dist = max(cnt_dist_from_center)
        smallest_dist = min(cnt_dist_from_center)
        median_dist = statistics.median(cnt_dist_from_center)

        blank_features = np.reshape(np.array((white_percent, len(contours), cnt_avg_size, median_cnt, largest_cnt,
                                              smallest_cnt, cnt_avg_dist, largest_dist, smallest_dist, median_dist))
                                    , (1, 10))
        predicted_blank = self.blank_model.predict_proba(blank_features)[0][0]
        return predicted_blank
