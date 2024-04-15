from calendar import c			#  ?
import cv2						# cv2 is the module import name for opencv-python
import numpy as np				# NumPy (Numerical Python) is an open source Python library that��s used in almost every field of science and engineering.
import setting					#  ?
import matplotlib.pyplot as plt


FUNCTIONS = [
    "get_processed_frame",
    "mask_marker",
    "marker_center",
    "inpaint",
    "difference",
    "get_all_contour",
    "get_convex_hull_area",
    "draw_flow",
]
CLASS = ["ContactArea"]

def get_processed_frame(frame):
    # Check if the frame is valid
    if frame is None or frame.size == 0:
        print("Received an invalid frame.")
        return None

    # Resize the frame to 800x600 if it's not already that size
    if frame.shape[1] != 800 or frame.shape[0] != 600:
        frame = cv2.resize(frame, (800, 600))

    # plt.figure("original")
    # plt.imshow(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Rotation and Downsampling
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    downsampled = cv2.pyrDown(rotated_frame).astype(np.uint8)	# Coverting cv2.pyrDown(rotated_frame) from float32(probably) to unsigned int 8

    return downsampled


def mask_marker(frame, debug=False):
    # Downsampling and Type Conversion
    m, n = frame.shape[1], frame.shape[0]
    frame = cv2.pyrDown(frame).astype(
        np.float32
    )

    # plt.figure("pyrDown")
    # plt.imshow(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Difference of Gaussians (DoG)
    blur = cv2.GaussianBlur(frame, (25, 25), 0)
    blur2 = cv2.GaussianBlur(frame, (15, 15), 0)

    # plt.figure("blur_25*25")
    # plt.imshow(cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    #
    # plt.figure("blur_15*15")
    # plt.imshow(cv2.cvtColor(blur2.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    diff = blur - blur2
    # plt.figure("blurdiff")
    # plt.imshow(cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Thresholding and Mask Creation
    diff *= 20  # Arbitrary value

    diff[diff < 0.0] = 0.0
    diff[diff > 255.0] = 255.0

    # plt.figure("blurdiffAmp")
    # plt.imshow(cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # diff = cv2.GaussianBlur(diff, (5, 5), 0)

    THRESHOLD = 120
    mask_b = diff[:, :, 0] > THRESHOLD
    mask_g = diff[:, :, 1] > THRESHOLD
    mask_r = diff[:, :, 2] > THRESHOLD

    # plt.figure("mask_b")
    # plt.imshow(cv2.cvtColor(mask_b.astype(np.uint8) * 255, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.figure("mask_g")
    # plt.imshow(cv2.cvtColor(mask_g.astype(np.uint8) * 255, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.figure("mask_r")
    # plt.imshow(cv2.cvtColor(mask_r.astype(np.uint8) * 255, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
    # plt.figure("mask_rgb")
    # plt.imshow(cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    if debug:
        cv2.imshow("maskdiff", diff.astype(np.uint8))
        cv2.imshow("mask", mask.astype(np.uint8) * 255)

    # Final Mask Upsampling and Output
    mask = cv2.resize(mask.astype(np.uint8), (m, n))
    # plt.figure("mask_return")
    # plt.imshow(cv2.cvtColor(mask * 255, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    return mask * 255  # Dot is white
     


def marker_center(frame, debug=False):
    # Initialization
    areaThresh1 = 20
    areaThresh2 = 500
    centers = []

    # Marker Mask Creation
    mask = mask_marker(frame, debug=debug)

    # Contour Detection
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours[0]) < 5:  # if too little markers, then give up
        print("Too less markers detected: ", len(contours))
        return centers

    # Marker Centers Calculation
    for i, contour in enumerate(contours[0]):
        x, y, w, h = cv2.boundingRect(contour)
        AreaCount = cv2.contourArea(contour)
        if (
            AreaCount > areaThresh1
            and AreaCount < areaThresh2
            and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1
        ):
            t = cv2.moments(contour)
            mc = [t["m10"] / t["m00"], t["m01"] / t["m00"]]
            centers.append(mc)
            # cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, (0, 0, 255), 2, 6)

    return centers


def inpaint(frame):
    # Mask Creation
    mask = mask_marker(frame, debug=False)
    # plt.figure("mask_marker")
    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    # print(frame.shape)
    # print(mask.shape)

    # Inpainting
    frame_marker_removed = cv2.inpaint(
        frame, mask, 7, cv2.INPAINT_TELEA
    )  # Inpain the white area in the mask (aka the marker). The number is the pixel neighborhood radius

    return frame_marker_removed

def difference(frame, frame0, debug=False):
    # Calculate Difference
    normalized_lb = 0.5     # Normalized lower bound
    SF = 0.7                # Scale factor
    diff = (frame * 1.0 - frame0) / 255.0 + normalized_lb  # Diff in range 0,1
    # plt.figure("diff")
    # plt.imshow(cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Reduce noncontact differences
    diff[diff < normalized_lb] = (diff[diff < normalized_lb] - normalized_lb) * SF + normalized_lb
    diff[diff > 1] = 1
    diff[diff < 0] = 0
    # plt.figure("diff_reduce")
    # plt.imshow(cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Convert to uint8, Apply Threshold, Gray Conversion and Thresholding
    diff_uint8 = (diff * 255).astype(np.uint8)
    diff_uint8_before = diff_uint8.copy()

    # plt.figure("diff_uint8_before")
    # plt.imshow(cv2.cvtColor(diff_uint8_before.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    diff_uint8[diff_uint8 > 140] = 255
    diff_uint8[diff_uint8 <= 140] = 0
    # plt.figure("diff_uint8_threshold")
    # plt.imshow(cv2.cvtColor(diff_uint8.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Modified part start
    # Define a threshold ratio for color dominance
    dominance_ratio = 5
    epsilon = 1e-3

    # Split the image into its channel components
    blue_channel, green_channel, red_channel = cv2.split(diff_uint8)

    # Calculate the dominance for each color
    red_dominance = (red_channel / (np.maximum(blue_channel, epsilon)) > dominance_ratio) | (
            red_channel / (np.maximum(green_channel, epsilon)) > dominance_ratio)
    green_dominance = (green_channel / (np.maximum(red_channel, epsilon)) > dominance_ratio) | (
            green_channel / (np.maximum(blue_channel, epsilon)) > dominance_ratio)
    blue_dominance = (blue_channel / (np.maximum(red_channel, epsilon)) > dominance_ratio) | (
            blue_channel / (np.maximum(green_channel, epsilon)) > dominance_ratio)

    # Determine non-dominant areas (where no single channel is dominant)
    non_singlecolor_mask = ~(red_dominance | green_dominance | blue_dominance)

    # Convert boolean mask to uint8
    non_singlecolor_mask = non_singlecolor_mask.astype(np.uint8) * 255

    # Apply the mask to the image to retain only non-dominant areas
    filtered_image = cv2.bitwise_and(diff_uint8, diff_uint8, mask=non_singlecolor_mask)

    # plt.figure("filtered_image")
    # plt.imshow(cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    # Modified part end

    diff_gray = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # plt.figure("diff_gray")
    # plt.imshow(cv2.cvtColor(diff_gray.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # diff_gray = 255 - diff_gray

    _, diff_thresh = cv2.threshold(
        diff_gray, 50, 255, cv2.THRESH_BINARY
    )  # Return 2 values, the second is the thresholded image

    # plt.figure("diff_graythresh")
    # plt.imshow(cv2.cvtColor(diff_thresh.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Erosion and Dilation
    diff_thresh_erode = cv2.erode(diff_thresh, np.ones((5, 5), np.uint8), iterations=2)
    # plt.figure("diff_thresh_erode")
    # plt.imshow(cv2.cvtColor(diff_thresh_erode.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    diff_thresh_dilate = cv2.dilate(
        diff_thresh_erode, np.ones((5, 5), np.uint8), iterations=1
    )
    # plt.figure("diff_thresh_dilate")
    # plt.imshow(cv2.cvtColor(diff_thresh_dilate.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    if debug: # Debugging Visualization
        cv2.imshow("diff_uint8", diff_uint8_before)
        cv2.imshow("diff_uint8 after", diff_uint8)
        cv2.imshow("diff", diff.astype(np.uint8))
        cv2.imshow("diff_gray", diff_gray)
        cv2.imshow("diff_graythresh", diff_thresh)
        cv2.imshow("diff_thresh_erode", diff_thresh_erode)
        cv2.imshow("diff_thresh_dilate", diff_thresh_dilate)

    return diff_thresh_dilate


def get_all_contour(diff_thresh_dilate, frame, debug=False):
    # Finding Contours
    contours, hierarchy = cv2.findContours(
        diff_thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Merging and Ellipse Fitting
    try:
        merged_contour = np.concatenate(contours)
        print("Merged contour: ", merged_contour.shape)
        ellipse = cv2.fitEllipse(merged_contour)

        # Modified part start
        # Scale factor for the size of the ellipse
        scale_factor = 1.3
        (center, axes, angle) = ellipse
        scaled_axes = (int(axes[0] * scale_factor), int(axes[1] * scale_factor))

        # Create the scaled ellipse tuple
        scaled_ellipse = (center, scaled_axes, angle)

        img_ellipse = frame.copy()
        contour_ellipse = cv2.cvtColor(diff_thresh_dilate.copy(), cv2.COLOR_GRAY2BGR)
        # Modified part end

        cv2.ellipse(img_ellipse, scaled_ellipse, (0, 255, 0), 2)
        cv2.ellipse(contour_ellipse, scaled_ellipse, (0, 255, 0), 2)

        if debug: # Debugging Visualization
            cv2.imshow("Ellipse", img_ellipse)
            cv2.imshow("Ellipse on Contour", contour_ellipse)
    except:
        print("No Contact found")
        pass
    return contours

def regress_line(all_points, frame, debug=False):
    # Line Frame Preparation
    line_frame = frame  # Draw on top of the original frame

    # Line Fitting
    vx, vy, x, y = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculating Slope and Angle
    slope = vy / vx
    angle = np.degrees(np.arctan(slope))

    # Determining Line Endpoints for Drawing
    lefty = int((-x * vy / vx) + y)
    righty = int(((line_frame.shape[1] - x) * vy / vx) + y)
    pt1 = (line_frame.shape[1] - 1, righty)
    pt2 = (0, lefty)

    # Calculating Midpoint
    midx = int((pt1[0] + pt2[0]) / 2)
    midy = int((pt1[1] + pt2[1]) / 2)

    # Drawing on the Frame
    cv2.line(line_frame, pt1, pt2, (0, 255, 0), 2)
    cv2.circle(line_frame, (int(midx), int(midy)), 5, (0, 255, 0), 2, 6)
    cv2.putText(
        line_frame,
        f"mid x: {midx}, mid y: {midy}",
        (5, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        line_frame,
        f"angle: {-1 *angle}",
        (5, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    if debug: # Debugging Visualization
        cv2.imshow("Line", line_frame)
    return -1*angle, (midy, midx)


def get_convex_hull_area(diff_thresh_dilate, frame, debug=False):
    # Find Contours
    contours, hierarchy = cv2.findContours(
        diff_thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize Variables
    img_hull = frame.copy()
    hull_area = 0
    slope = None
    center = None
    hull_mask = np.zeros(diff_thresh_dilate.shape, dtype=np.uint8)

    # Loop Over Contours
    if len(contours) > 0:
        try:
            # Create a single array of points from all contours
            contourpoints = np.vstack([cnt for cnt in contours])
            # Convex Hull Calculation
            hullpoints = cv2.convexHull(contourpoints, returnPoints=True)
            # cv2.drawContours(img_hull, [hullpoints], -1, (0, 255, 0), 2)
            hull_area = cv2.contourArea(hullpoints)

            # Line Regression
            slope, center = regress_line(contourpoints, img_hull, debug=True)

            # Create Hull Mask
            cv2.fillPoly(hull_mask, pts=[hullpoints], color=(255, 255, 255))
        except Exception as e:
            print("Hull", e)
            pass

    if debug: # Debugging Visualization
        cv2.putText(
            img_hull,
            f"Hull Area: {hull_area}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Convex Hull", img_hull)
        cv2.imshow("Hull Mask", hull_mask)

    return hull_area, hull_mask, slope, center


def draw_flow(frame, flow):
    # Inputs
    Ox, Oy, Cx, Cy, Occupied = flow

    # Calculating arrow vectors
    K = 1
    drawn_frame = frame.copy()
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            # pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt1 = (int(Cx[i][j]), int(Cy[i][j]))

            pt2 = (
                int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])),
                int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])),
            )
            color = (0, 255, 255) # Color Coding
            if Occupied[i][j] <= -1:
                color = (255, 255, 255)

            # Drawing Arrows
            cv2.arrowedLine(drawn_frame, pt1, pt2, color, 2, tipLength=0.2)

    # plt.figure("drawn_frame")
    # plt.imshow(cv2.cvtColor(drawn_frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    return drawn_frame


def draw_flow_mask(frame, flow, mask, debug=False):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 2

    # Frame Preparation
    drawn_frame = frame.copy()

    # Combining Mask and Frame
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    # plt.figure("mask")
    # plt.imshow(cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    # Combining Mask and Frame
    drawn_frame_and = cv2.bitwise_and(drawn_frame, drawn_frame, mask=mask)

    # plt.figure("drawn_frame_and")
    # plt.imshow(cv2.cvtColor(drawn_frame_and.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    change = [0, 0]
    counter = 0

    # Flow Visualization Within Masked Area
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            # pt1 = (int(Ox[i][j]), int(Oy[i][j]))

            if mask[int(Cy[i][j]), int(Cx[i][j])] == 255:
                dx = int(Cx[i][j] - Ox[i][j])
                dy = int(Cy[i][j] - Oy[i][j])
                pt1 = (int(Cx[i][j]), int(Cy[i][j]))

                pt2 = (
                    int(Cx[i][j] + K * dx),
                    int(Cy[i][j] + K * dy),
                )
                # Average Flow Calculation
                counter += 1
                change[0] += dx
                change[1] += dy
                color = (0, 255, 255)
                if Occupied[i][j] <= -1:
                    color = (255, 255, 255)

                cv2.arrowedLine(drawn_frame_and, pt1, pt2, color, 2, tipLength=0.2)
    if counter > 0:
        change[0] /= counter
        change[1] /= counter
        change_text = f"Average: [{change[0]:.3f}, {change[1]:.3f}]"
        cv2.putText(
            drawn_frame_and,
            change_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if debug:
        cv2.imshow("Flow Hull", mask)

    # plt.figure("drawn_frame_and")
    # plt.imshow(cv2.cvtColor(drawn_frame_and.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)
    return drawn_frame_and, change


# This class is not used
class ContactArea:
    def __init__(
        self, base=None, draw_poly=True, contour_threshold=100, *args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        (
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
        ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            try:
                self._draw_major_minor(
                    target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
                )
                print("Drawn")
            except Exception as e:
                print("Error drawing major/minor axis: ", e)
                pass
        return (major_axis, major_axis_end), (minor_axis, minor_axis_end)

    def _diff(self, target, base):
        diff = (target * 1.0 - base) / 255.0 + 0.5

        cv2.imshow("diff1", diff)
        # print("Diff range", np.min(diff), np.max(diff))

        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5

        cv2.imshow("diff2", diff)
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        cv2.imshow("Diff_Abs", diff_abs)

        return diff_abs

    def _smooth(self, target):
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        cv2.imshow("Diff_Blur", diff_blur)
        return diff_blur

    def _contours(self, target):
        mask = ((np.abs(target) > 0.08) * 255).astype(np.uint8)  # Default > 0.04
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel)

        cv2.imshow("Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
        self,
        target,
        poly,
        major_axis,
        major_axis_end,
        minor_axis,
        minor_axis_end,
        lineThickness=2,
    ):
        poly = None
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        poly = None
        major_axis = []
        major_axis_end = []
        minor_axis = []
        minor_axis_end = []

        for contour in contours:
            if len(contour) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end
