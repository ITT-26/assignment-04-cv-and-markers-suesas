import argparse
import os

import cv2
import numpy as np


PREVIEW_WINDOW = "Image Extractor"
RESULT_WINDOW = "Warped Result"

original_img = None
warped_img = None
selected_points = []
args = None
result_window_open = False


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a clicked image region with perspective transformation.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, required=True, help="Output width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Output height in pixels")
    return parser.parse_args()


def order_points(points):
    points = np.array(points, dtype=np.float32)

    # Order corners for getPerspectiveTransform: top-left, top-right, bottom-right, bottom-left.
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_diffs)]
    ordered[3] = points[np.argmax(point_diffs)]
    return ordered


def draw_preview():
    preview_img = original_img.copy()
    for index, point in enumerate(selected_points):
        cv2.circle(preview_img, point, 5, (255, 0, 0), -1)
        cv2.putText(
            preview_img,
            str(index + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    if len(selected_points) > 1:
        cv2.polylines(preview_img, [np.array(selected_points, dtype=np.int32)], False, (0, 255, 0), 2)

    cv2.imshow(PREVIEW_WINDOW, preview_img)


def make_warp():
    src = order_points(selected_points)
    dst = np.array(
        [
            [0, 0],
            [args.width - 1, 0],
            [args.width - 1, args.height - 1],
            [0, args.height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(original_img, matrix, (args.width, args.height))


def mouse_callback(event, x, y, flags, param):
    global warped_img, result_window_open

    if event != cv2.EVENT_LBUTTONDOWN or len(selected_points) >= 4:
        return

    selected_points.append((x, y))
    draw_preview()

    if len(selected_points) == 4:
        warped_img = make_warp()
        cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(RESULT_WINDOW, warped_img)
        result_window_open = True


def reset_selection():
    global warped_img, result_window_open

    selected_points.clear()
    warped_img = None
    if result_window_open:
        try:
            cv2.destroyWindow(RESULT_WINDOW)
        except cv2.error:
            pass
        result_window_open = False
    draw_preview()


def save_result():
    if warped_img is None:
        print("Select four points before saving.")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print("Output directory does not exist:", output_dir)
        return

    if cv2.imwrite(args.output, warped_img):
        print("Saved:", args.output)
    else:
        print("Could not save:", args.output)


def main():
    global args, original_img

    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Width and height must be positive.")

    original_img = cv2.imread(args.input)
    if original_img is None:
        raise FileNotFoundError("Could not load image: " + args.input)

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(PREVIEW_WINDOW, mouse_callback)
    draw_preview()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            reset_selection()
        elif key == ord("s") or key == ord("S"):
            save_result()
        else:
            try:
                if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
