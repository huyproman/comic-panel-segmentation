"""Source-faithful extraction of Pipeline.ipynb stages into canonical historical pipeline."""
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any
from scipy.stats import mode
from sklearn.cluster import MeanShift

from .config import HistoricalLegacyConfig


class LegacyLine:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, img_w: int = 384, img_h: int = 384):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.orientation = self._set_orientation()
        # Historical Pipeline.ipynb Cell 3: self.list_dis = self.dis_from_ref(384, 384)
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

    def cross_angle(self, other: 'LegacyLine') -> float:
        dx1, dy1 = self.x2 - self.x1, self.y2 - self.y1
        dx2, dy2 = other.x2 - other.x1, other.y2 - other.y1
        m1, m2 = math.sqrt(dx1 ** 2 + dy1 ** 2), math.sqrt(dx2 ** 2 + dy2 ** 2)
        if m1 * m2 == 0:
            return 0.0
        dot = dx1 * dx2 + dy1 * dy2
        cos_v = max(min(dot / (m1 * m2), 1.0), -1.0)
        return float(np.degrees(np.arccos(cos_v)) % 360)

    def _dis_from_ref(self, w: int, l: int) -> List[float]:
        p1 = (w * 5.0 / 7.0, l * 0.5)
        p2 = (w / 6.0, l * 6.0 / 7.0)
        p3 = (w * 3.5 / 9.0, l / 3.0)

        def dis_pl(x0: float, y0: float) -> float:
            tu = abs((self.y2 - self.y1) * x0 - (self.x2 - self.x1) * y0 + self.x2 * self.y1 - self.x1 * self.y2)
            mau = math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
            return float(tu / mau) if mau > 0 else 0.0

        return [dis_pl(*p1), dis_pl(*p2), dis_pl(*p3)]


def smooth_edges(mask: np.ndarray, kernel_size: Tuple[int, int] = (20, 20)) -> np.ndarray:
    """Cell 5: smooth_edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morphing_edges(image: np.ndarray) -> np.ndarray:
    """Cell 6: morphing applied to Canny edges."""
    dilation_kernel_1 = np.ones((5, 5), np.uint8)
    erosion_kernel_1 = np.ones((5, 5), np.uint8)
    erosion_kernel_2 = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(image, dilation_kernel_1, iterations=2)
    eroded_image_2 = cv2.erode(dilated_image, erosion_kernel_1, iterations=1)
    processed_image = cv2.erode(eroded_image_2, erosion_kernel_2, iterations=1)
    return processed_image


def group_lines_historical(lines_list: List[LegacyLine], ang_tol: float = 3.0, dis_tol: float = 8.5) -> List[List[LegacyLine]]:
    """Cell 9: group_lines."""
    groups = []
    for i in range(len(lines_list)):
        grp = [lines_list[i]]
        for j in range(i + 1, len(lines_list)):
            d_ang = lines_list[i].cross_angle(lines_list[j])
            d_dis = np.array(lines_list[i].list_dis) - np.array(lines_list[j].list_dis)
            if (d_ang < ang_tol or d_ang > (180.0 - ang_tol)) and np.all(np.abs(d_dis) < dis_tol):
                grp.append(lines_list[j])
        if grp:
            groups.append(grp)
    return groups


def merge_groups_historical(list_of_lists1: List[List[LegacyLine]]) -> List[LegacyLine]:
    """Cell 10: merge_lists using mode/median angle selection (NO geometric line rotation)."""
    merged = []
    list_of_lists = list_of_lists1.copy()
    while list_of_lists:
        cur = list_of_lists.pop(0)
        i = 0
        while i < len(list_of_lists):
            common = any(l in list_of_lists[i] for l in cur)
            if common:
                cur.extend(list_of_lists.pop(i))
            else:
                i += 1
        merged.append(list(set(cur)))

    final_m = []
    for lset in merged:
        d = {line: line.calculate_angle() for line in lset}
        vals = list(d.values())
        m_res = mode(vals, keepdims=True)
        if m_res.count[0] >= 2:
            final_m.extend([k for k, v in d.items() if v == m_res.mode[0]])
        else:
            med = sorted(vals)[len(vals) // 2]
            final_m.extend([k for k, v in d.items() if v == med])
    return final_m


def calc_inter(l1: LegacyLine, l2: LegacyLine) -> Tuple[int, int] | None:
    """Cell 14: intersection calculation."""
    x1, y1, x2, y2 = l1.x1, l1.y1, l1.x2, l1.y2
    x3, y3, x4, y4 = l2.x1, l2.y1, l2.x2, l2.y2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)


def bresenham_points(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """Cell 20: bresenham."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    points = []
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def is_connection_valid_historical(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, threshold: float = 0.75) -> bool:
    """Cell 19: is_connection_valid."""
    line_pts = bresenham_points(x1, y1, x2, y2)
    if not line_pts:
        return False
    colored_count = 0
    h, w = img.shape[:2]
    for x, y in line_pts:
        if 0 <= x < w and 0 <= y < h:
            if img[y, x] > 20:
                colored_count += 1
    return (colored_count / len(line_pts)) >= threshold


def run_canonical_historical_postprocessing(raw_mask: np.ndarray, config: HistoricalLegacyConfig = HistoricalLegacyConfig()) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute the full source-faithful legacy Pipeline.ipynb workflow."""
    h, w = raw_mask.shape[:2]

    # Cell 5: smooth_edges
    og_smoothed = smooth_edges(raw_mask, config.morphology.mask_opening_kernel_size)
    og_smoothed_copy = og_smoothed.copy()

    # Cell 22: Canny
    edges_1 = cv2.Canny(og_smoothed_copy, threshold1=config.morphology.canny_threshold1, threshold2=config.morphology.canny_threshold2)

    # Cell 6: morphing on edges
    testing_1 = morphing_edges(edges_1)

    # Cell 22: Hough
    lines1 = cv2.HoughLinesP(
        testing_1,
        config.hough.rho,
        config.hough.theta_degrees * np.pi / 180.0,
        threshold=config.hough.threshold,
        minLineLength=int(config.hough.min_line_length),
        maxLineGap=int(config.hough.max_line_gap)
    )
    if lines1 is None:
        return og_smoothed_copy, {"centroids_count": 0, "connections": 0}

    v_lines, h_lines = [], []
    for l in lines1:
        line_obj = LegacyLine(*l[0], img_w=w, img_h=h)
        if line_obj.orientation == 'vertical':
            v_lines.append(line_obj)
        else:
            h_lines.append(line_obj)

    fin_h = merge_groups_historical(group_lines_historical(h_lines, ang_tol=config.grouping.angle_tolerance, dis_tol=config.grouping.distance_tolerance))
    fin_v = merge_groups_historical(group_lines_historical(v_lines, ang_tol=config.grouping.angle_tolerance, dis_tol=config.grouping.distance_tolerance))

    # Cell 8: create_image_with_classified_lines
    hough_classified1 = np.zeros_like(raw_mask)
    for l in (fin_h + fin_v):
        cv2.line(hough_classified1, (l.x1, l.y1), (l.x2, l.y2), 255, 2)

    intersections = [calc_inter(vl, hl) for vl in fin_v for hl in fin_h]
    intersections = [p for p in intersections if p is not None]

    border_lines = [
        LegacyLine(0, 0, w - 1, 0, img_w=w, img_h=h),
        LegacyLine(0, h - 1, w - 1, h - 1, img_w=w, img_h=h),
        LegacyLine(0, 0, 0, h - 1, img_w=w, img_h=h),
        LegacyLine(w - 1, 0, w - 1, h - 1, img_w=w, img_h=h)
    ]

    # Legacy clipping: [0, 400]
    clip_max = config.reconstruction.coordinate_clip_max
    v_border = [calc_inter(vl, bl) for vl in fin_v for bl in border_lines]
    v_border = [p for p in v_border if p is not None and all(0 <= c <= clip_max for c in p)]

    h_border = [calc_inter(hl, bl) for hl in fin_h for bl in border_lines]
    h_border = [p for p in h_border if p is not None and all(0 <= c <= clip_max for c in p)]

    border_inter = v_border + h_border

    centroids = []
    if len(intersections) > 0:
        ms_in = MeanShift(bandwidth=config.reconstruction.inner_bandwidth, bin_seeding=True)
        ms_in.fit(np.unique(np.array(intersections), axis=0))
        centroids.extend([tuple(int(x) for x in c) for c in ms_in.cluster_centers_])
    if len(border_inter) > 0:
        ms_b = MeanShift(bandwidth=config.reconstruction.border_bandwidth, bin_seeding=True)
        ms_b.fit(np.unique(np.array(border_inter), axis=0))
        centroids.extend([tuple(int(x) for x in c) for c in ms_b.cluster_centers_])

    # Cell 18: draw_centroid_connection WITH is_connection_valid
    ve_len = og_smoothed_copy.copy()
    image1copy = hough_classified1.copy()
    conns_drawn = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            p1, p2 = centroids[i], centroids[j]
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            if is_connection_valid_historical(image1copy, x1, y1, x2, y2, threshold=config.reconstruction.support_threshold):
                if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > config.reconstruction.min_euclidean_distance:
                    cv2.line(ve_len, (x1, y1), (x2, y2), config.reconstruction.line_color, config.reconstruction.line_thickness)
                    conns_drawn += 1

    diag = {
        "centroids_count": len(centroids),
        "validated_connections": conns_drawn,
        "raw_shape": (h, w)
    }
    return ve_len, diag
