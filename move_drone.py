import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import Twist

import time


class DroneMover(Node):
    def __init__(self):
        super().__init__("drone_mover")

        self.takeoff_pub = self.create_publisher(Empty, "/simple_drone/takeoff", 10)
        self.vel_mode_pub = self.create_publisher(Bool, "/simple_drone/dronevel_mode", 10)
        self.posctrl_pub = self.create_publisher(Bool, "/simple_drone/posctrl", 10)
        self.cmd_pub = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)

    def publish_for(self, pub, msg, duration=1.0):
        end_time = time.time() + duration
        while time.time() < end_time:
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)

    def run(self):
        time.sleep(1)

        # takeoff
        self.publish_for(self.takeoff_pub, Empty(), 2.0)

        # velocity mode ON
        vel_mode = Bool()
        vel_mode.data = True
        self.publish_for(self.vel_mode_pub, vel_mode, 1.0)

        # position control OFF
        posctrl = Bool()
        posctrl.data = False
        self.publish_for(self.posctrl_pub, posctrl, 1.0)

        # move up
        msg = Twist()
        msg.linear.z = 1.0
        self.publish_for(self.cmd_pub, msg, 3.0)

        # move forward
        msg = Twist()
        msg.linear.x = 1.0
        self.publish_for(self.cmd_pub, msg, 3.0)

        # stop
        msg = Twist()
        self.publish_for(self.cmd_pub, msg, 1.0)


def main():
    rclpy.init()
    node = DroneMover()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()