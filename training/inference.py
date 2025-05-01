#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time 

class YOLOv8Node:
    def __init__(self):
        # ── ROS setup ────────────────────────────────────────────────────────────
        rospy.init_node("yolov8_inference_node")
        repo_path = Path(__file__).resolve().parent.parent
        # Parameters  (set in a launch file or with rosparam)
        weights_path = repo_path / "training" / "models" / "jasmine.pt"
        device       = 0
        conf_thres   = 0.25

        # ── Model ───────────────────────────────────────────────────────────────
        rospy.loginfo("[YOLOv8] loading model %s on %s…", weights_path, device)
        self.model   = YOLO(weights_path)
        self.model.fuse()                      # slight speed‑up
        self.model.to(device)

        # ── Image transport ─────────────────────────────────────────────────────
        self.bridge  = CvBridge()
        self.sub     = rospy.Subscriber("/camera/color/image_raw",
                                        Image, self.callback,
                                        queue_size=1, buff_size=2**24)
        self.pub     = rospy.Publisher("/yolov8/annotated",
                                       Image, queue_size=1)

        self.conf_thres = conf_thres
        rospy.loginfo("[YOLOv8] node ready – waiting for images…")

    # ────────────────────────────────────────────────────────────────────────────
    def callback(self, msg: Image) -> None:
        """Run YOLO on incoming ROS Image and publish / display result."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.flip(frame, -1)
        except CvBridgeError as err:
            rospy.logerr("cv_bridge error: %s", err)
            return

        t0 = time.perf_counter()    
        result = self.model(frame, conf=self.conf_thres, verbose=False)[0]
        t1 = time.perf_counter()               # ── end timer ──
        dt_ms  = (t1 - t0) * 1_000             # milliseconds
        fps    = 1.0 / (t1 - t0) if t1 != t0 else 0.0
        rospy.loginfo_throttle(1.0,
            "YOLOv8 latency: %.1f ms  (%.1f FPS)", dt_ms, fps)
        annotated = result.plot()

        cv2.imshow("YOLOv8 Detections", annotated)
        cv2.waitKey(1)

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header          # preserve timestamp/frame‑id
        self.pub.publish(out)

    def spin(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        YOLOv8Node().spin()
    except rospy.ROSInterruptException:
        pass
