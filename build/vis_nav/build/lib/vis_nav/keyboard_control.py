import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import tty
import termios

msg = """
Control Your Robot!
---------------------------
Moving around:
        z
   q    s    d
        x

w/x : increase/decrease linear velocity
a/d : increase/decrease angular velocity
space key : force stop

CTRL-C to quit
"""

class TeleKey(Node):
    def __init__(self):
        super().__init__('telekey')
        self.twist = Twist()
        self.pub = self.create_publisher(Twist, '/scout/cmd_vel', 10)
        self.create_subscription(Twist, '/scout/cmd_vel', self.cmd_callback, 10)

        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.backup_linear_vel = 0.0
        self.backup_angular_vel = 0.0
        self.linear_vel_limit = 0.5
        self.angular_vel_limit = 0.6
        self.flag = False

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def vels(self):
        return f"currently:\tlinear vel {self.target_linear_vel:.2f}\t angular vel {self.target_angular_vel:.2f}"

    def cmd_callback(self, msg):
        self.backup_linear_vel = msg.linear.x
        self.backup_angular_vel = msg.angular.z
        # self.get_logger().info(
        #     f"Backup velocities updated: linear={self.backup_linear_vel}, angular={self.backup_angular_vel}"
        # )

    def control_loop(self):
        global settings
        settings = termios.tcgetattr(sys.stdin)
        print(msg)

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == '1':
                    self.target_linear_vel = self.backup_linear_vel
                    self.target_angular_vel = self.backup_angular_vel
                    self.twist.angular.x = 0.0
                    self.flag = True
                    self.get_logger().info('Engage!!!')
                elif key == '2':
                    self.twist.angular.x = 0.0
                    self.flag = False
                    self.get_logger().info('DisEngage!!!')
                elif key == '\x03':  # CTRL+C
                    break

                if self.flag:
                    self.update_velocities(key)

                # Clamp velocities
                # Clamp velocities
                self.target_linear_vel = max(0.0, min(self.target_linear_vel, self.linear_vel_limit))

                # self.target_linear_vel = max(
                #     min(self.target_linear_vel, self.linear_vel_limit), -self.linear_vel_limit
                # )
                self.target_angular_vel = max(
                    min(self.target_angular_vel, self.angular_vel_limit), -self.angular_vel_limit
                )

                self.twist.linear.x = self.target_linear_vel
                
                self.twist.angular.z = self.target_angular_vel
                # self.get_logger().info(
                #             f"Backup velocities updated: linear={self.twist.linear.x}, angular={self.twist.angular.z} \nself.twist.linear.y = {self.twist.linear.y} \nself.twist.linear.z = {self.twist.linear.z}"                      )
                self.pub.publish(self.twist)
        finally:
            self.stop_robot()

    def update_velocities(self, key):
        if key == 'z':
            self.target_linear_vel += 0.05
        elif key == 's':
            self.target_linear_vel -= 0.05
        elif key == 'q':
            self.target_angular_vel += 0.05
        elif key == 'd':
            self.target_angular_vel -= 0.05
        elif key == 'x':
            self.target_linear_vel = 0.0
            self.target_angular_vel = 0.0
        elif key == 'a':
            self.target_angular_vel = 0.0
        elif key == ' ':
            self.target_linear_vel = -1.0
            self.target_angular_vel = 0.0

        # self.get_logger().info(self.vels())

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    rclpy.init(args=args)
    telekey = TeleKey()

    try:
        telekey.control_loop()
    except KeyboardInterrupt:
        telekey.get_logger().info('Node interrupted by user')
    finally:
        telekey.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
