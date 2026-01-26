import numpy as np
import math
import matplotlib.pyplot as plt
# from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_control.robots.robot import BaseRobot
from safe_control.utils import plotting, env
from position_control.occlusion_cbf_qp import OcclusionCBF

class StoppingBackupController:
    """Simple backup controller that decelerates to a stop."""
    def compute_control(self, x, target=None):
        # x is [pos_x, pos_y, vel_x, vel_y]
        # u = -Kp * v
        v = x[2:4].flatten()
        u = -2.0 * v 
        return u

def main():
    # 1. Setup Environment
    # Simple open space with some obstacles
    known_obs = np.array([
        [8.0, 5.0, 1.0],
        [8.0, 8.0, 1.0]
    ])
    
    # Pad to 7 columns as required by env.py
    if len(known_obs) > 0 and known_obs.shape[1] != 7:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 4))))
    
    env_width = 15.0
    env_height = 15.0
    
    plt.ion() # Enable interactive mode
    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("Occlusion CBF Test")
    
    # 2. Setup Robot
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'v_max': 2.0,
        'a_max': 2.0,
        'radius': 0.5
    }
    dt = 0.05
    x_init = np.array([2.0, 2.0, 0.0, 0.0])
    
    # BaseRobot requires 5 states for DoubleIntegrator2D [x, y, vx, vy, theta]
    x_init_base = np.append(x_init, 0.0).reshape(-1, 1)
    
    # Use BaseRobot wrapper which supports visualization
    base_robot = BaseRobot(x_init_base, robot_spec, dt, ax)
    
    # 3. Setup Controller
    # OcclusionCBF needs the stateless inner robot for predictions
    controller = OcclusionCBF(
        robot=base_robot.robot, 
        robot_spec=robot_spec,
        dt=dt,
        backup_horizon=2.0,
        ax=ax
    )
    
    # Setup nominal controller (simple P controller to goal)
    goal = np.array([12.0, 12.0])
    
    def nominal_control(x):
        pos = x[:2].flatten()
        vel = x[2:4].flatten()
        
        # PD control to goal
        kp = 1.0
        kd = 1.0
        
        err = goal - pos
        u = kp * err - kd * vel
        
        # Clamp controls
        u = np.clip(u, -robot_spec['a_max'], robot_spec['a_max'])
        return u
        
    controller.set_nominal_controller(nominal_control)
    
    # Setup backup controller
    backup_ctrl = StoppingBackupController()
    controller.set_backup_controller(backup_ctrl)
    
    # Setup environment for CBF
    # Create valid environment object for BackupCBF
    # It expects objects with 'obstacles' attribute
    class SimpleEnv:
        def __init__(self, obstacles):
            self.obstacles = []
            for obs in obstacles:
                self.obstacles.append({'x': obs[0], 'y': obs[1], 'radius': obs[2]})
            self.half_width = 100.0 # Open space
            
    cbf_env = SimpleEnv(known_obs)
    controller.set_environment(cbf_env)
    
    # 4. Run Simulation
    tf = 100
    steps = int(tf / dt)
    
    current_state = x_init.copy()
    
    print("Starting simulation...")
    for i in range(steps):
        # 1. Get control
        # BaseRobot.X is (4, 1), flatten for controller
        current_state = base_robot.X.flatten()
        u = controller.solve_control_problem(current_state)
        u = u.flatten()
        
        # 2. Step robot
        # BaseRobot.step updates internal state and handles plotting internals
        base_robot.step(u)
        
        # 3. Visualization
        if i % 1 == 0:  # Plot every step for smoother animation
            base_robot.render_plot()
            # plot_handler.plot_goal(goal, 0.5) # Assuming plot_goal might not exist or needs verification
            # Manual goal plotting if needed
            ax.scatter(goal[0], goal[1], c='g', s=50, label='Goal')
            
            # Flush events to update plot
            # fig.canvas.flush_events()
            plt.pause(0.001)
            
        # Check if reached
        dist = np.linalg.norm(current_state[:2] - goal)
        if dist < 0.5:
            print(f"Goal reached at step {i}")
            break
            
    print("Simulation finished.")
    plt.ioff()
    plt.show(block=True)

if __name__ == '__main__':
    main()
