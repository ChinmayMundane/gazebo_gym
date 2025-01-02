import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class StopRobot(Node):
    def __init__(self):
        super().__init__('stop_robot')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.stop_robot()
            
    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher_.publish(stop_msg)
        self.get_logger().info('Stopping the robot...')
        
def main(args=None):
    rclpy.init(args=args)
    node = StopRobot()
    rclpy.spin_once(node, timeout_sec=1)  # Allow the node to publish the stop command
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()