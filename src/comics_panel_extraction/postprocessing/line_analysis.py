import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.stats import mode
from sklearn.cluster import MeanShift
@dataclass(frozen=True)
class PostProcessParameters:
    opening_kernel_size: Tuple[int, int] = (20, 20)
    canny_thresh1: int = 50
    canny_thresh2: int = 100
    hough_rho: float = 2.0
    hough_theta: float = np.pi / 180.0
    hough_threshold: int = 25
    hough_min_line_length: float = 10.0
    hough_max_line_gap: float = 30.0
    angle_threshold_deg: float = 3.0
    distance_threshold_px: float = 8.5
    inner_bandwidth: float = 13.0
    border_bandwidth: float = 10.0
    coordinate_clip_max: int = 400
    bresenham_support_threshold: float = 0.75
    min_euclidean_distance: float = 40.0
    line_color: int = 0
    line_thickness: int = 3
    size_threshold: int = 1000
class DetectedLine:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, img_w: int = 384, img_h: int = 384):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.orientation = self._set_orientation()
        self.list_dis = self._dis_from_ref(384, 384)
    def calculate_angle(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        mag = math.sqrt(dx ** 2 + dy ** 2)
        if mag == 0:
            return 0.0
        cos_val = max(min(dx / mag, 1.0), -1.0)
        return float(np.degrees(np.arccos(cos_val)) % 360)
    def _set_orientation(self) -> str:
        ang = self.calculate_angle()
        if ang > 180:
            ang -= 180
        if ang > 90:
            ang = 180 - ang
        return 'vertical' if ang >= 45 else 'horizontal'
    def cross_angle(self, other: 'DetectedLine') -> float:
        dx1, dy1 = self.x2 - self.x1, self.y2 - self.y1
        dx2, dy2 = other.x2 - other.x1, other.y2 - other.y1
        m1, m2 = math.sqrt(dx1 ** 2 + dy1 ** 2), math.sqrt(dx2 ** 2 + dy2 ** 2)
        if m1 * m2 == 0:
            return 0.0
        dot = dx1 * dx2 + dy1 * dy2
        cos_val = max(min(dot / (m1 * m2), 1.0), -1.0)
        return float(np.degrees(np.arccos(cos_val)))
    def _dis_from_ref(self, w: int, h: int) -> List[float]:
        ref_points = [
            (float(w) * 5.0 / 7.0, float(h) * 0.5),
            (float(w) / 6.0, float(h) * 6.0 / 7.0),
            (float(w) * 3.5 / 9.0, float(h) / 3.0)
        ]
        x1, y1 = float(self.x1), float(self.y1)
        x2, y2 = float(self.x2), float(self.y2)
        denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if denom == 0.0:
            return [math.sqrt((px - x1) ** 2 + (py - y1) ** 2) for px, py in ref_points]
        distances = []
        for px, py in ref_points:
            num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
            distances.append(float(num / denom))
        return distances
def calc_inter(line1: DetectedLine, line2: DetectedLine) -> Optional[Tuple[int, int]]:
    x1, y1, x2, y2 = line1.x1, line1.y1, line1.x2, line1.y2
    x3, y3, x4, y4 = line2.x1, line2.y1, line2.x2, line2.y2
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return int(round(x)), int(round(y))
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points
def is_connection_valid(image_edge: np.ndarray, x1: int, y1: int, x2: int, y2: int, threshold: float = 0.75) -> bool:
    h, w = image_edge.shape[:2]
    points = bresenham_line(x1, y1, x2, y2)
    val_points = [p for p in points if 0 <= p[0] < w and 0 <= p[1] < h]
    if not val_points:
        return False
    support = sum(1 for p in val_points if image_edge[p[1], p[0]] > 0)
    return (support / len(val_points)) >= threshold
def filter_components_by_size(mask: np.ndarray, size_threshold: int = 1000) -> np.ndarray:
    bin_m = (mask > 127).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(bin_m, connectivity=8)
    filtered = np.zeros_like(bin_m)
    for lbl in range(1, num_labels):
        comp_m = (labels == lbl).astype(np.uint8)
        if cv2.countNonZero(comp_m) > size_threshold:
            filtered[labels == lbl] = 255
    return filtered
def run_line_segment_postprocessing(
    raw_mask: np.ndarray,
    params: Optional[PostProcessParameters] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if params is None:
        params = PostProcessParameters()
    mask_255 = (raw_mask > 127).astype(np.uint8) * 255 if raw_mask.max() > 1 else (raw_mask > 0).astype(np.uint8) * 255
    h, w = mask_255.shape[:2]
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.opening_kernel_size)
    og_smoothed = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, open_kernel)
    edges = cv2.Canny(og_smoothed, params.canny_thresh1, params.canny_thresh2, apertureSize=3)
    kernel5 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    bridged = cv2.dilate(edges, kernel5, iterations=2)
    bridged = cv2.erode(bridged, kernel5, iterations=1)
    bridged = cv2.erode(bridged, kernel3, iterations=1)
    lines_raw = cv2.HoughLinesP(
        bridged,
        rho=params.hough_rho,
        theta=params.hough_theta,
        threshold=params.hough_threshold,
        minLineLength=params.hough_min_line_length,
        maxLineGap=params.hough_max_line_gap
    )
    detected_lines = []
    if lines_raw is not None:
        for l in lines_raw:
            x1, y1, x2, y2 = l[0]
            detected_lines.append(DetectedLine(x1, y1, x2, y2, img_w=w, img_h=h))
    horizontals = [l for l in detected_lines if l.orientation == 'horizontal']
    verticals = [l for l in detected_lines if l.orientation == 'vertical']
    def group_lines(line_list: List[DetectedLine]) -> List[DetectedLine]:
        if not line_list:
            return []
        groups: List[List[DetectedLine]] = []
        for l in line_list:
            assigned = False
            for grp in groups:
                rep = grp[0]
                cross = l.cross_angle(rep)
                angle_ok = (cross < params.angle_threshold_deg) or (cross > (180.0 - params.angle_threshold_deg))
                if angle_ok:
                    dist_diffs = [abs(a - b) for a, b in zip(l.list_dis, rep.list_dis)]
                    if max(dist_diffs) <= params.distance_threshold_px:
                        grp.append(l)
                        assigned = True
                        break
            if not assigned:
                groups.append([l])
        merged = []
        for grp in groups:
            x1_m = int(round(np.mean([l.x1 for l in grp])))
            y1_m = int(round(np.mean([l.y1 for l in grp])))
            x2_m = int(round(np.mean([l.x2 for l in grp])))
            y2_m = int(round(np.mean([l.y2 for l in grp])))
            merged.append(DetectedLine(x1_m, y1_m, x2_m, y2_m, img_w=w, img_h=h))
        return merged
    fin_h = group_lines(horizontals)
    fin_v = group_lines(verticals)
    inner_inter = [calc_inter(hl, vl) for hl in fin_h for vl in fin_v]
    inner_inter = [p for p in inner_inter if p is not None and (0 <= p[0] <= params.coordinate_clip_max) and (0 <= p[1] <= params.coordinate_clip_max)]
    border_lines = [
        DetectedLine(0, 0, w - 1, 0, img_w=w, img_h=h),
        DetectedLine(0, h - 1, w - 1, h - 1, img_w=w, img_h=h),
        DetectedLine(0, 0, 0, h - 1, img_w=w, img_h=h),
        DetectedLine(w - 1, 0, w - 1, h - 1, img_w=w, img_h=h)
    ]
    v_border = [calc_inter(vl, bl) for vl in fin_v for bl in border_lines]
    v_border = [p for p in v_border if p is not None and (0 <= p[0] <= params.coordinate_clip_max) and (0 <= p[1] <= params.coordinate_clip_max)]
    h_border = [calc_inter(hl, bl) for hl in fin_h for bl in border_lines]
    h_border = [p for p in h_border if p is not None and (0 <= p[0] <= params.coordinate_clip_max) and (0 <= p[1] <= params.coordinate_clip_max)]
    border_inter = v_border + h_border
    centroids = []
    if len(inner_inter) > 0:
        ms_in = MeanShift(bandwidth=params.inner_bandwidth, bin_seeding=True)
        ms_in.fit(np.unique(np.array(inner_inter), axis=0))
        centroids.extend([tuple(int(x) for x in c) for c in ms_in.cluster_centers_])
    if len(border_inter) > 0:
        ms_b = MeanShift(bandwidth=params.border_bandwidth, bin_seeding=True)
        ms_b.fit(np.unique(np.array(border_inter), axis=0))
        centroids.extend([tuple(int(x) for x in c) for c in ms_b.cluster_centers_])
    carved = og_smoothed.copy()
    conns_drawn = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            p1, p2 = centroids[i], centroids[j]
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist > params.min_euclidean_distance:
                if is_connection_valid(bridged, x1, y1, x2, y2, threshold=params.bresenham_support_threshold):
                    cv2.line(carved, (x1, y1), (x2, y2), params.line_color, params.line_thickness)
                    conns_drawn += 1
    final_mask = filter_components_by_size(carved, size_threshold=params.size_threshold)
    final_mask_uint8 = (final_mask > 127).astype(np.uint8) * 255
    diagnostics = {
        "detected_lines_count": len(detected_lines),
        "horizontal_merged_count": len(fin_h),
        "vertical_merged_count": len(fin_v),
        "centroids_count": len(centroids),
        "validated_connections": conns_drawn,
    }
    return final_mask_uint8, diagnostics
