import os
import cv2
import numpy as np
import find_marker
import A_utility_test
import setting
import copy
import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt


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


image_folder = 'Testdata'  # Replace with the dataset path
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]  # Assuming images are in PNG format


count = 0
frame0 = None


for image_file in sorted(image_files):
    print("Processing file:", image_file)

    # Camera Frame Capture
    frame = cv2.imread(os.path.join(image_folder, image_file))
    # plt.figure("original_frame")
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)  # Pause for a short period to allow the window to update
    if frame is None:
        print("Failed to load image:", image_file)
        continue

    # Frame Preprocessing
    count += 1
    frame = A_utility_test.get_processed_frame(frame)
    # plt.figure("get_processed_frame")
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)  # Pause for a short period to allow the window to update

    # Contact Area Calculation
    if count == 1:
        frame0 = copy.deepcopy(frame)
        frame0_final = A_utility_test.inpaint(frame0)
        # plt.figure("inpaint frame0")
        # plt.imshow(cv2.cvtColor(frame0_final, cv2.COLOR_BGR2RGB))
        # plt.show(block=False)
        # plt.pause(0.001)

    frame_final = A_utility_test.inpaint(frame)
    # plt.figure("inpaint")
    # plt.imshow(cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB))
    # plt.show(block=False)
    # plt.pause(0.001)

    # Save the image to the workspace
    # output_path1 = '/home/chengjin/Projects/Gelsight_codes/VM/Intermidiate output/frame_final.png'
    # cv2.imwrite(output_path1, frame_final)
    # output_path2 = '/home/chengjin/Projects/Gelsight_codes/VM/Intermidiate output/frame0_final.png'
    # cv2.imwrite(output_path2, frame0_final)
    #
    # print(f"Image saved to {output_path2}")

    contact_area_dilated = A_utility_test.difference(frame_final, frame0_final, debug=False)
    # output_path3 = '/home/chengjin/Projects/Gelsight_codes/VM/Intermidiate output/contact_area_dilated.png'
    # cv2.imwrite(output_path3, contact_area_dilated)

    # Convex Hull Analysis
    contours = A_utility_test.get_all_contour(contact_area_dilated, frame, debug=False)
    hull_area, hull_mask,slope, center = A_utility_test.get_convex_hull_area(
        contact_area_dilated, frame, debug=True
    )  # Hull area and slope

    # Marker Detection and Optical Flow Analysis
    m_centers = A_utility_test.marker_center(frame, debug=False)
    m.init(m_centers)
    m.run()
    flow = m.get_flow()  # FLOW

    # Flow Visualization
    frame_flow = A_utility_test.draw_flow(frame, flow)
    frame_flow_hull, average_flow_change_in_hull = A_utility_test.draw_flow_mask(
        frame, flow, hull_mask, debug=False
    )

    # Display and Data Publishing
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
