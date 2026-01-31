import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped

class DepthImageSaver(Node):
    def __init__(self):
        super().__init__('depth_image_saver')
        self.i = 0
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            cv_image_normalized = cv_image_normalized.astype(np.uint8)

            # Save the normalized image as a .png file
            if self.i < 2:
                cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav/results/depth_image_{self.i}.png', cv_image_normalized)
                self.get_logger().info(f'Depth image saved as depth_image_{self.i}.png')
                self.i += 1
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))
class GoalPose_subscriber(Node):
    def __init__(self):
        super().__init__('GoalPose_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            1)
        self.subscription

    def goal_pose_callback(self, data):
        global goal_pose_rviz
        goal_pose_rviz = data
        self.get_logger().info(f'Data is {goal_pose_rviz.pose.position.x}')
class Image_fish_subscriber(Node):
    def __init__(self):
        super().__init__('image_fish_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera2/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, rgb_data):
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_data, "mono8")
            #original_image = self.br.imgmsg_to_cv2(rgb_data, "rgb8")
            last_image_fish = image[80:400, 140:500]
            #image_ = self.br.imgmsg_to_cv2(rgb_data, "rgb8")

            cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav/results/feshEye_imageCropped.png', last_image_fish)
            self.get_logger().info(f'Depth image saved as feshEye_imageCropped.png')
            cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav/results/feshEye_image.png', image)
            self.get_logger().info(f'Depth image saved as feshEye_image.png')

            
        except Exception as e:
            self.get_logger().error('Could not convert fish image: %s' % str(e))

def main(args=None):
    rclpy.init(args=args)
    node = GoalPose_subscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

