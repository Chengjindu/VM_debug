import cv2
import numpy as np
import find_marker
import A_utility_live_mjpg
import setting
import copy
import roslib
import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point

cap = cv2.VideoCapture("http://10.255.32.36:8080/?action=stream") # Replace this with the pi address
# ret, frame = cap.read()

# Load the transformation matrix
M = np.load('transformation_matrix.npy')
width, height = 800, 600

def publish_image(real_x, real_y, real_z):
    detect_result = Point()
    rate = rospy.Rate(100)
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'

    detect_result.x = real_x
    detect_result.y = real_y
    detect_result.z = real_z
    pub.publish(detect_result)


rospy.init_node('tactile_senser')
pub = rospy.Publisher("/tactile", Point, queue_size=10)

## Init marker tracking components
setting.init()
m = find_marker.Matching(  # Instance of th find_marker library
    N_=setting.N_,
    M_=setting.M_,
    fps_=setting.fps_,
    x0_=setting.x0_,
    y0_=setting.y0_,
    dx_=setting.dx_,
    dy_=setting.dy_,
)

count = 0
frame0 = None

while True:
    ret, frame = cap.read()  # 'ret' is True if a frame is captured successfully
    if not ret or frame is None:
        continue  # Skip the loop iteration if no frame is captured

    # Apply the perspective transformation
    if 'M' in locals():  # Check if M exists
        frame = cv2.warpPerspective(frame, M, (width, height))
    else:
        print("Transformation matrix M is not defined.")

    count += 1

    frame = A_utility_live_mjpg.get_processed_frame(frame)  # Get processed frame from UDP stream
    if frame is None:
        continue  # Skip further processing if the frame is invalid

    if count == 1:
        frame0 = copy.deepcopy(frame)
        frame0_final = A_utility_live_mjpg.inpaint(frame0)

    frame_final = A_utility_live_mjpg.inpaint(frame)
    # frame_final = frame

    contact_area_dilated = A_utility_live_mjpg.difference(frame_final, frame0_final, debug=False)
    contours = A_utility_live_mjpg.get_all_contour(contact_area_dilated, frame, debug=False)
    hull_area, hull_mask,slope, center = A_utility_live_mjpg.get_convex_hull_area(
        contact_area_dilated, frame, debug=True
    )  # Hull area and slope

    ## Marker
    m_centers = A_utility_live_mjpg.marker_center(frame, debug=False)
    m.init(m_centers)
    m.run()
    flow = m.get_flow()  # FLOW

    frame_flow = A_utility_live_mjpg.draw_flow(frame, flow)
    frame_flow_hull, average_flow_change_in_hull = A_utility_live_mjpg.draw_flow_mask(
        frame, flow, hull_mask, debug=False
    )

    # Show frame
    if True:
        cv2.imshow("frame_flow", frame_flow)

        if  slope is not None:
            publish_image(average_flow_change_in_hull[0],average_flow_change_in_hull[1], slope)
        else:
            publish_image(1, 2, 0)

    # End loop, print FPS
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()  # Release the video capture object
