#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class OpenArmVisionNode(Node):
    def __init__(self):
        super().__init__("openarm_vision_node")
        
        # 1. 配置参数
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/camera_info")
        
        # 2. 图像转换器
        self.bridge = CvBridge()
        
        # 3. 发布者
        self.banana_pub = self.create_publisher(PoseStamped, "/banana_pose", 10)
        self.debug_pub = self.create_publisher(Image, "/vision_debug", 10)
        
        # 4. 订阅者 (使用时间同步，同时获取RGB和深度图)
        rgb_sub = message_filters.Subscriber(self, Image, self.get_parameter("camera_topic").value)
        depth_sub = message_filters.Subscriber(self, Image, self.get_parameter("depth_topic").value)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)
        
        # 相机内参 (通常从 camera_info 获取，这里先给默认值防止未收到 info)
        self.camera_intrinsics = None
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value, self.info_callback, 10)
        
        self.get_logger().info("视觉节点已启动，等待图像数据...")

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(f"相机内参已接收: {self.camera_intrinsics}")

    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_intrinsics is None:
            return

        try:
            # 1. 转换 ROS 图像 -> OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            # 2. 图像处理 (HSV 检测)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 香蕉阈值 (黄色)
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 轮廓提取
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            best_banana_pos = None
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500: # 过滤噪点
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # 绘制矩形框
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    cx = x + w // 2
                    cy = y + h // 2
                    
                    # 3. 获取深度并计算 3D 坐标
                    # 注意：深度图可能是 float (米) 或 uint16 (毫米)
                    d_val = cv_depth[cy, cx]
                    
                    # 简单判断深度单位
                    if np.max(cv_depth) > 100: # 可能是毫米
                        d_val = d_val / 1000.0
                        
                    if d_val > 0.1 and d_val < 2.0: # 有效深度范围
                        # 坐标反投影 (Pinhole Model)
                        # Z = d
                        # X = (u - cx) * Z / fx
                        # Y = (v - cy) * Z / fy
                        
                        Z = float(d_val)
                        X = (cx - self.camera_intrinsics['cx']) * Z / self.camera_intrinsics['fx']
                        Y = (cy - self.camera_intrinsics['cy']) * Z / self.camera_intrinsics['fy']
                        
                        best_banana_pos = (X, Y, Z)
                        
                        label = f"Banana: {X:.2f}, {Y:.2f}, {Z:.2f}"
                        cv2.putText(cv_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 4. 发布结果
            if best_banana_pos:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = rgb_msg.header.frame_id # 使用相机的坐标系
                
                pose_msg.pose.position.x = best_banana_pos[0]
                pose_msg.pose.position.y = best_banana_pos[1]
                pose_msg.pose.position.z = best_banana_pos[2]
                
                # 默认姿态
                pose_msg.pose.orientation.w = 1.0
                
                self.banana_pub.publish(pose_msg)
                # self.get_logger().info(f"发布香蕉坐标: {best_banana_pos}")
                
            # 发布调试图像
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            
        except Exception as e:
            self.get_logger().error(f"图像处理出错: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OpenArmVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
