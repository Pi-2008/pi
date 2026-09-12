"""
Hershey's Chocolate Spread Coverage Detector
=============================================

Measures what percentage of a fixed rectangular "bread area" is covered by
Hershey's chocolate spread, using simple HSV color segmentation (no ML).

Run:
    python main.py

See README.md for full setup, camera positioning, and calibration steps.
"""

import json
import os

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

# =============================================================================
# CONFIGURATION - tweak these to match your setup
# =============================================================================

CONFIG_FILE = "config.json"
CAMERA_INDEX = 1

# --- Chocolate color detection (HSV) ---
# Starting guess for Hershey's chocolate spread under normal indoor lighting.
# Use calibration mode (press C, then 2) to dial these in for your camera,
# then copy the printed values back in here so they're the defaults next time.
CHOCOLATE_LOWER = (0, 60, 20)      # H, S, V lower bound
CHOCOLATE_UPPER = (30, 255, 180)   # H, S, V upper bound

# --- Bread rectangle (used only until you calibrate; then config.json takes over) ---
DEFAULT_RECT = {"x": 150, "y": 100, "w": 300, "h": 300}  # pixels: top-left + size

# Shrinks the rectangle inward on all sides so the physical border/marking
# itself never gets counted as "chocolate."
RECTANGLE_MARGIN = 10

# --- Image processing ---
BLUR_KERNEL = (5, 5)
MORPH_KERNEL = np.ones((5, 5), np.uint8)
MORPH_OPEN_ITERATIONS = 2
MORPH_CLOSE_ITERATIONS = 2

# --- Optional uniformity metric ---
UNIFORMITY_GRID = 4  # splits the bread area into a 4x4 grid of cells

# --- "Perfect coverage" audio alert ---
PERFECT_COVERAGE_THRESHOLD = 90.0   # percent; play the sound at/above this
PERFECT_SOUND_FILE = "perfect.mp3"  # must sit next to this script (or give a full path)

# --- Coverage-tier rating image ---
# Checked top-down; first (threshold, filename) whose threshold the coverage
# meets or exceeds wins. Must end with a 0 entry so every percentage matches.
TIER_IMAGES = [
    (85.0, "1.png"),   # 85-100%
    (50.0, "2.png"),   # 50-84.999...%
    (0.0, "3.png"),    # 0-49.999...%
]

# --- Home screen ---
HOME_IMAGE_FILE = "10.png"
HOME_TITLE_TEXT = "CHECK YOUR SPREAD"

# =============================================================================
# CONFIG PERSISTENCE
# =============================================================================

def load_config():
    """Load saved rectangle/HSV calibration, or fall back to the defaults above."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            print(f"[config] Loaded saved calibration from {CONFIG_FILE}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] Could not read {CONFIG_FILE} ({e}); using defaults.")
    return {
        "rect": dict(DEFAULT_RECT),
        "hsv_lower": list(CHOCOLATE_LOWER),
        "hsv_upper": list(CHOCOLATE_UPPER),
    }


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved calibration to {CONFIG_FILE}")


# =============================================================================
# AUDIO ALERT
# =============================================================================

# Resolve perfect.mp3 relative to this script's own folder, not the process's
# current working directory - so it's found no matter where you run `python`
# from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PERFECT_SOUND_PATH = os.path.join(_SCRIPT_DIR, PERFECT_SOUND_FILE)

_mixer_ready = False
if pygame is not None:
    try:
        pygame.mixer.init()
        _mixer_ready = True
    except Exception as e:
        print(f"[audio] Could not initialize pygame mixer: {e}")


def play_perfect_sound():
    """
    Play PERFECT_SOUND_FILE via pygame.mixer. Playback is non-blocking by
    nature (mixer.music.play() returns immediately), so it never freezes the
    camera loop. Safe to call even if pygame or the mp3 file isn't available -
    it just logs and skips.
    """
    if pygame is None:
        print("[audio] 'pygame' package not installed - run `pip install pygame` "
              "to enable the perfect-coverage sound.")
        return

    if not _mixer_ready:
        print("[audio] pygame mixer failed to initialize - check your audio device/drivers.")
        return

    if not os.path.exists(_PERFECT_SOUND_PATH):
        print(f"[audio] '{PERFECT_SOUND_FILE}' not found at {_PERFECT_SOUND_PATH} - skipping sound.")
        return

    try:
        # Only restart playback if it isn't already playing this sound, so
        # rapid re-triggers don't stutter/cut themselves off.
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(_PERFECT_SOUND_PATH)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"[audio] Failed to play {PERFECT_SOUND_FILE}: {e}")


# =============================================================================
# COVERAGE-TIER RATING IMAGE
# =============================================================================

def get_tier_image_filename(coverage_pct):
    """Pick the rating image filename for a given coverage percentage."""
    for threshold, filename in TIER_IMAGES:
        if coverage_pct >= threshold:
            return filename
    return TIER_IMAGES[-1][1]  # fallback, shouldn't normally be hit


def show_tier_image(coverage_pct):
    """Load and display the rating image that matches this coverage tier,
    scaled to fill the screen without cropping or distorting it."""
    filename = get_tier_image_filename(coverage_pct)
    path = os.path.join(_SCRIPT_DIR, filename)

    if not os.path.exists(path):
        print(f"[image] '{filename}' not found at {path} - skipping rating image.")
        return

    img = cv2.imread(path)
    if img is None:
        print(f"[image] Could not read '{filename}' - is it a valid image file?")
        return

    cv2.namedWindow("Rating", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rating", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Figure out the actual fullscreen window size (varies per monitor), then
    # letterbox the image into it so the whole thing is visible - no cropping,
    # no stretching, just black bars on whichever side doesn't match.
    screen_w, screen_h = _get_screen_resolution()
    fitted = _letterbox_to_size(img, screen_w, screen_h)

    cv2.imshow("Rating", fitted)


def _get_screen_resolution():
    """Best-effort screen size lookup; falls back to a common 1080p size."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def _letterbox_to_size(img, target_w, target_h):
    """Resize `img` to fit within target_w x target_h, preserving aspect
    ratio, and pad the leftover space with black bars so the full image is
    always visible (no cropping) and never distorted."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# =============================================================================
# HOME SCREEN (title button gates entry into capture/retake/customize)
# =============================================================================

def build_home_screen(screen_w, screen_h):
    """
    Build the fullscreen home image (10.png, letterboxed) with a clickable
    title button drawn on top. Returns (canvas, button_rect) where
    button_rect = (x, y, w, h) in canvas pixel coordinates - used later to
    test whether a click landed on the title.
    """
    path = os.path.join(_SCRIPT_DIR, HOME_IMAGE_FILE)

    if os.path.exists(path):
        img = cv2.imread(path)
    else:
        img = None
        print(f"[home] '{HOME_IMAGE_FILE}' not found at {path} - using a black background instead.")

    if img is None:
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    else:
        canvas = _letterbox_to_size(img, screen_w, screen_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(HOME_TITLE_TEXT, font, scale, thickness)

    pad_x, pad_y = 45, 28
    btn_w = text_w + 2 * pad_x
    btn_h = text_h + 2 * pad_y
    btn_x = (screen_w - btn_w) // 2
    btn_y = int(screen_h * 0.78) - btn_h // 2  # sits about 3/4 down the screen

    # Semi-transparent dark button behind the title, so it reads clearly over
    # any part of the background image.
    overlay = canvas.copy()
    cv2.rectangle(overlay, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (30, 30, 30), -1)
    canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)
    cv2.rectangle(canvas, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)

    text_x = btn_x + pad_x
    text_y = btn_y + pad_y + text_h
    cv2.putText(canvas, HOME_TITLE_TEXT, (text_x, text_y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas, (btn_x, btn_y, btn_w, btn_h)


def point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def make_home_click_callback(button_rect, click_state):
    """
    Returns an OpenCV mouse callback that sets click_state["clicked"] = True
    only when a left-click lands inside button_rect (the title button) -
    clicks anywhere else on the home screen do nothing.
    """
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and point_in_rect(x, y, button_rect):
            click_state["clicked"] = True
    return on_mouse


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def clamp_rect_to_frame(rect, frame_shape):
    """Make sure the rectangle never falls outside the actual camera frame."""
    h, w = frame_shape[:2]
    x = max(0, min(rect["x"], w - 1))
    y = max(0, min(rect["y"], h - 1))
    rw = max(1, min(rect["w"], w - x))
    rh = max(1, min(rect["h"], h - y))
    return {"x": x, "y": y, "w": rw, "h": rh}


def get_inner_rect(rect, margin):
    """Shrink the rectangle by `margin` px on every side (excludes the border)."""
    x = rect["x"] + margin
    y = rect["y"] + margin
    w = max(1, rect["w"] - 2 * margin)
    h = max(1, rect["h"] - 2 * margin)
    return {"x": x, "y": y, "w": w, "h": h}


def crop_rect(image, rect):
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    return image[y:y + h, x:x + w]


def draw_rect_overlay(frame, rect, margin):
    """Draw the outer marked rectangle (blue) and the inner measured area (green)."""
    out = frame.copy()
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)  # outer = blue

    inner = get_inner_rect(rect, margin)
    ix, iy, iw, ih = inner["x"], inner["y"], inner["w"], inner["h"]
    cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)  # inner = green
    return out


# =============================================================================
# CHOCOLATE DETECTION
# =============================================================================

def build_chocolate_mask(bgr_image, hsv_lower, hsv_upper):
    """Return a binary mask: white = detected chocolate, black = not chocolate."""
    blurred = cv2.GaussianBlur(bgr_image, BLUR_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # Remove small noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_OPEN_ITERATIONS)
    # Fill small holes inside chocolate regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_CLOSE_ITERATIONS)
    return mask


def compute_uniformity(mask, grid_n):
    """
    Simple secondary metric: splits the bread area into a grid and looks at
    how consistent the coverage is cell-to-cell. 100 = perfectly even spread,
    lower = patchy/uneven. This is a rough heuristic, not a precise measure.
    """
    h, w = mask.shape[:2]
    if h < grid_n or w < grid_n:
        return 100.0  # area too small to grid meaningfully

    cell_h, cell_w = h // grid_n, w // grid_n
    cell_percentages = []
    for row in range(grid_n):
        for col in range(grid_n):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h if row < grid_n - 1 else h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w if col < grid_n - 1 else w
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            cell_percentages.append(100.0 * cv2.countNonZero(cell) / cell.size)

    if not cell_percentages:
        return 100.0

    std_dev = float(np.std(cell_percentages))
    uniformity = max(0.0, 100.0 - std_dev)
    return uniformity


# =============================================================================
# ANALYSIS + DISPLAY
# =============================================================================

def analyze_and_show(frame, config):
    rect = clamp_rect_to_frame(config["rect"], frame.shape)
    inner = get_inner_rect(rect, RECTANGLE_MARGIN)

    bread_area = crop_rect(frame, inner)
    if bread_area.size == 0:
        print("[error] Bread rectangle is empty/invalid - recalibrate with 'C' then '1'.")
        return

    mask = build_chocolate_mask(bread_area, config["hsv_lower"], config["hsv_upper"])

    total_pixels = bread_area.shape[0] * bread_area.shape[1]
    chocolate_pixels = cv2.countNonZero(mask)
    coverage_pct = 100.0 * chocolate_pixels / total_pixels if total_pixels else 0.0
    uniformity_pct = compute_uniformity(mask, UNIFORMITY_GRID)

    is_perfect = coverage_pct >= PERFECT_COVERAGE_THRESHOLD
    if is_perfect:
        play_perfect_sound()

    show_tier_image(coverage_pct)

    # --- Build the overlay: highlight detected chocolate in semi-transparent red ---
    red_layer = np.full_like(bread_area, (0, 0, 255))  # BGR red
    blended = cv2.addWeighted(bread_area, 0.4, red_layer, 0.6, 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
    overlay = np.where(mask_3ch, blended, bread_area).astype(np.uint8)

    # --- Original captured frame, with rectangle drawn for reference ---
    original_annotated = draw_rect_overlay(frame, rect, RECTANGLE_MARGIN)

    # --- Results text panel ---
    panel = np.zeros((220, 520, 3), dtype=np.uint8)
    cv2.putText(panel, f"HERSHEY'S COVERAGE: {coverage_pct:.1f}%", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Chocolate pixels: {chocolate_pixels:,}", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Total bread-area pixels: {total_pixels:,}", (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Uniformity (experimental): {uniformity_pct:.1f}%", (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if is_perfect:
        cv2.putText(panel, "PERFECT COVERAGE!", (15, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Press R to go back live, Q to quit", (15, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow("Original (Captured)", original_annotated)
    cv2.imshow("Chocolate Mask", mask)
    cv2.imshow("Overlay", overlay)
    cv2.imshow("Results", panel)

    print(f"\nHERSHEY'S COVERAGE: {coverage_pct:.1f}%")
    print(f"Chocolate pixels: {chocolate_pixels:,}")
    print(f"Total bread-area pixels: {total_pixels:,}")
    print(f"Uniformity (experimental): {uniformity_pct:.1f}%")
    if is_perfect:
        print(f"[audio] Coverage >= {PERFECT_COVERAGE_THRESHOLD:.0f}% - playing {PERFECT_SOUND_FILE}")


def close_result_windows():
    for name in ("Original (Captured)", "Chocolate Mask", "Overlay", "Results", "Rating"):
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


# =============================================================================
# CALIBRATION: BREAD RECTANGLE (click two corners)
# =============================================================================

def calibrate_rectangle(cap, config):
    """
    Click the TOP-LEFT then BOTTOM-RIGHT corner of the marked rectangle.
    Press ENTER to confirm, R to redo the two clicks, ESC to cancel.
    """
    print("\n[calibrate-rect] Click TOP-LEFT then BOTTOM-RIGHT corner of the "
          "marked rectangle. ENTER=confirm, R=redo, ESC=cancel.")

    points = []
    window = "Calibrate Rectangle - click TL then BR, ENTER=confirm"
    cv2.namedWindow(window)

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback(window, on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[error] Could not read from camera.")
            break

        display = frame.copy()
        for pt in points:
            cv2.drawMarker(display, pt, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        if len(points) == 2:
            cv2.rectangle(display, points[0], points[1], (0, 255, 0), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("[calibrate-rect] Cancelled.")
            break
        elif key in (ord('r'), ord('R')):
            points.clear()
        elif key == 13 and len(points) == 2:  # ENTER
            (x1, y1), (x2, y2) = points
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w < 2 * RECTANGLE_MARGIN + 5 or h < 2 * RECTANGLE_MARGIN + 5:
                print("[calibrate-rect] Rectangle too small, try again.")
                points.clear()
                continue
            config["rect"] = {"x": x, "y": y, "w": w, "h": h}
            save_config(config)
            print(f"[calibrate-rect] Saved rectangle: {config['rect']}")
            break

    cv2.destroyWindow(window)


# =============================================================================
# CALIBRATION: HSV THRESHOLDS (trackbars)
# =============================================================================

def calibrate_hsv(cap, config):
    """
    Live trackbars over the bread-area crop so you can dial in HSV thresholds
    for your actual Hershey's spread and lighting. Press S to save, ESC to exit.
    """
    print("\n[calibrate-hsv] Adjust trackbars until the mask cleanly covers just "
          "the chocolate. Press S to save, ESC to exit without saving.")

    window = "HSV Calibration (S=save, ESC=exit)"
    cv2.namedWindow(window)

    lo = config["hsv_lower"]
    hi = config["hsv_upper"]

    cv2.createTrackbar("H Min", window, lo[0], 179, lambda v: None)
    cv2.createTrackbar("H Max", window, hi[0], 179, lambda v: None)
    cv2.createTrackbar("S Min", window, lo[1], 255, lambda v: None)
    cv2.createTrackbar("S Max", window, hi[1], 255, lambda v: None)
    cv2.createTrackbar("V Min", window, lo[2], 255, lambda v: None)
    cv2.createTrackbar("V Max", window, hi[2], 255, lambda v: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[error] Could not read from camera.")
            break

        rect = clamp_rect_to_frame(config["rect"], frame.shape)
        inner = get_inner_rect(rect, RECTANGLE_MARGIN)
        bread_area = crop_rect(frame, inner)
        if bread_area.size == 0:
            print("[error] Bread rectangle invalid - calibrate rectangle first ('C' then '1').")
            break

        h_min = cv2.getTrackbarPos("H Min", window)
        h_max = cv2.getTrackbarPos("H Max", window)
        s_min = cv2.getTrackbarPos("S Min", window)
        s_max = cv2.getTrackbarPos("S Max", window)
        v_min = cv2.getTrackbarPos("V Min", window)
        v_max = cv2.getTrackbarPos("V Max", window)
        current_lower = (h_min, s_min, v_min)
        current_upper = (h_max, s_max, v_max)

        mask = build_chocolate_mask(bread_area, current_lower, current_upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Resize both to the same height so they can sit side by side
        target_h = 300
        scale = target_h / bread_area.shape[0]
        original_r = cv2.resize(bread_area, None, fx=scale, fy=scale)
        mask_r = cv2.resize(mask_bgr, None, fx=scale, fy=scale)
        combined = np.hstack([original_r, mask_r])

        cv2.imshow(window, combined)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("[calibrate-hsv] Exited without saving.")
            break
        elif key in (ord('s'), ord('S')):
            config["hsv_lower"] = list(current_lower)
            config["hsv_upper"] = list(current_upper)
            save_config(config)
            print("[calibrate-hsv] Saved. Copy these into main.py if you want them as defaults:")
            print(f"  CHOCOLATE_LOWER = {current_lower}")
            print(f"  CHOCOLATE_UPPER = {current_upper}")
            break

    cv2.destroyWindow(window)


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    config = load_config()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[error] Could not open camera index {CAMERA_INDEX}. "
              "Check the connection or try a different CAMERA_INDEX in main.py.")
        return

    print("\n=== Hershey's Chocolate Spread Coverage Detector ===")
    print("Click the title on the home screen to begin.")
    print("SPACE = capture & analyze | R = back to live | C = calibrate | Q = quit\n")

    # --- HOME SCREEN: nothing else opens until the title button is clicked ---
    screen_w, screen_h = _get_screen_resolution()
    home_img, button_rect = build_home_screen(screen_w, screen_h)
    home_click_state = {"clicked": False}

    cv2.namedWindow("Home", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Home", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Home", make_home_click_callback(button_rect, home_click_state))

    while True:
        cv2.imshow("Home", home_img)
        key = cv2.waitKey(20) & 0xFF
        if home_click_state["clicked"]:
            break
        if key in (ord('q'), ord('Q')):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Home")

    state = "LIVE"

    while True:
        if state == "LIVE":
            ret, frame = cap.read()
            if not ret:
                print("[error] Lost camera feed.")
                break

            rect = clamp_rect_to_frame(config["rect"], frame.shape)
            display = draw_rect_overlay(frame, rect, RECTANGLE_MARGIN)
            cv2.putText(display, "SPACE:Capture  R:Reset  C:Calibrate  Q:Quit",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Live Camera", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                analyze_and_show(frame, config)
                state = "RESULT"
            elif key in (ord('c'), ord('C')):
                print("\n[calibrate] Press 1 = calibrate rectangle, 2 = calibrate HSV, "
                      "any other key = cancel.")
                sub_key = cv2.waitKey(0) & 0xFF
                if sub_key == ord('1'):
                    calibrate_rectangle(cap, config)
                elif sub_key == ord('2'):
                    calibrate_hsv(cap, config)
                else:
                    print("[calibrate] Cancelled.")
            elif key in (ord('q'), ord('Q')):
                break

        elif state == "RESULT":
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('r'), ord('R')):
                close_result_windows()
                state = "LIVE"
            elif key in (ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
