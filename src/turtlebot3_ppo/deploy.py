import rclpy
from turtlebot3_env import TurtleBot3Env
from ppo_agent import PPOAgent
import torch
from train import reset_gazebo_world
import time
from stop_robot import *
import rclpy
import matplotlib.pyplot as plt

def main(args=None):
    rclpy.init(args=args)

    # Stop the robot by running the stop_robot script
    stop = StopRobot()
    stop.stop_robot()
    
    reset_gazebo_world() # Reset the Gazebo world
    env = TurtleBot3Env()
    agent = PPOAgent(env)
    agent.actor_critic.load_state_dict(torch.load('ppo_model.pth'))

    state = env.reset()
    time.sleep(2) # wait for the simulation to be ready
    
    robot_position = []
    
    while True:
        action, _ = agent.select_action(state)
        state, _, done, _ = env.step(action)
        robot_position.append((state[-5], state[-4]))
        if done:
            break
    env.close()
    rclpy.shutdown()
    
    # Stop the robot by running the stop_robot script
    stop.stop_robot()

    # Extract x and y coordinates for plotting
    x_coords = [pos[0] for pos in robot_position]
    y_coords = [pos[1] for pos in robot_position]

    # Plotting the path
    plt.plot(x_coords, y_coords, color='blue')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.title('TurtleBot3 Path')
    plt.show()
    
if __name__ == '__main__':
    main()
