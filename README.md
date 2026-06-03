# Hand Controller
 
ROS 2 (Humble) package that wraps [hand_tracking](https://github.com/DwnNyxDev/hand_tracking) into a node. Subscribes to a webcam image topic, runs hand landmark detection, and publishes `AckermannDriveStamped` commands based on finger gestures.

Used to add gesture-based teleoperation to [MIT RSS's simulated racecar](https://github.com/hcr-vvatel/hand_controlled_racecar_docker), a fork of [mit-rss/racecar_docker](https://github.com/mit-rss/racecar_docker)

## Current Gesture Controls
 
| Fingers Raised | Command |
|---|---|
| 0 | Go straight |
| 1 (index only) | Turn — angle maps to index finger direction |
| 2 | Accelerate |
| 3 | Decelerate |
| 5 | Stop |
 
## Build & Run
 
```bash
cd ~/your_ws/src
git clone https://github.com/hcr-vvatel/hand_controller.git
cd .. && colcon build --packages-select hand_controller
source install/setup.bash
ros2 run hand_controller hand_controller
```

## Used In
 
- [racecar_docker](https://github.com/hcr-vvatel/hand_controlled_racecar_docker) — gesture-controlled teleoperation of MIT RSS's simulated racecar in Gazebo.
