#!/usr/bin/env python

import rospy
import time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from math import sqrt

# Define plant colors
COLOR_RANGES = {
    "Tomato": ([355, 74, 65], [5, 74, 65]),   # Red
    "Pepper": ([55, 82, 99], [64, 82, 99]), # Yellow
    "Eggplant": ([255, 80, 95], [265, 80, 95]) # Purple
}

# Define indexed points
BASE_POINTS = [
    # Same points as before...
]

def create_extended_points(base_points):
    extended_points = []
    for p in base_points:
        extended_points.append(p)
        extended_points.append((p[0] + 2.8, p[1], p[2]))
        extended_points.append((p[0] - 2.8, p[1], p[2]))
    return extended_points

EXTENDED_POINTS = []

bridge = CvBridge()
current_position = (0, 0, 0)
plant_type = ""
total_objects = 0
tracked_points = []

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def plants_beds_callback(msg):
    global plant_type, tracked_points, EXTENDED_POINTS
    data = msg.data.split()
    plant_type = data[0]
    indices = list(map(int, data[1:]))
    base_selected_points = [BASE_POINTS[i] for i in indices if i < len(BASE_POINTS)]
    EXTENDED_POINTS = create_extended_points(base_selected_points)
    tracked_points = EXTENDED_POINTS

def odometry_callback(msg):
    global current_position
    current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
    check_uav_position()

def check_uav_position():
    for point in tracked_points:
        distance = sqrt((current_position[0] - point[0])**2 + (current_position[1] - point[1])**2 + (current_position[2] - point[2])**2)
        if distance < 0.5:
            rospy.loginfo("Taking picture at tracked point")
            take_picture()

def take_picture():
    rospy.wait_for_message("/red/camera/color/image_raw", Image)
    image_msg = rospy.Subscriber("/red/camera/color/image_raw", Image, image_callback)

def image_callback(img_msg):
    global total_objects
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Loop over all detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can change the confidence threshold
                center_x = int(detection[0] * cv_image.shape[1])
                center_y = int(detection[1] * cv_image.shape[0])
                w = int(detection[2] * cv_image.shape[1])
                h = int(detection[3] * cv_image.shape[0])

                # Draw rectangle and label for detected objects
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maximum Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            total_objects += 1

    # Display results
    cv2.imshow("YOLO Detection", cv_image)
    cv2.waitKey(1)

def camera_feed_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def monitor_uav():
    rospy.init_node('uav_monitor', anonymous=True)
    rospy.Subscriber("/red/plants_beds", String, plants_beds_callback)
    rospy.Subscriber("/red/odometry", Odometry, odometry_callback)
    rospy.Subscriber("/red/camera/color/image_raw", Image, camera_feed_callback)
    rospy.spin()

    if current_position == (1, 1, 1):
        rospy.loginfo(f"Quantidade total de frutos: {total_objects}")

if __name__ == '__main__':
    try:
        monitor_uav()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()



