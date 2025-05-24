# Fuzzy-Adaptive-Compliance-Control
Speed and Trajectory Optimization for Safe Human Robot Collaboration

## requirements：

envirionment:

    ubuntu 20.04
    ros noetic (ros melodic is not supported)
    python3.x

you should have the low-level control interface of your robot and adjust the communication of pid_executer
In this case kortex kinova gen3 is used

TODO: remove the coupled code of gen3 plattform
    
## Installation：

### get the repo
    mkdir  -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    git clone https://github.com/HeinrichvonAnonym/Fuzzy-Adaptive-Compliance-Control

notice! hardware drivers and interfaces is not included, you should install them by yourself

### build
    # install ros noetic firstly
    cd ..
    catkin_make
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

##   Usage
### modify:
you should modify the keys in ./src/speed_plan/config/dynamical_parameters.yaml:

    hard_ware_jointstate_topic: ${YOUR_HARD_WARE_JOINTSTATE_TOPIC}
    control_input_name: ${YOUR_CONTROL_INPUT_NAME}
    control_interface_class: ${YOUR_CONTROL_INTERFACE_CLASS}
    principle is as following:
    | THIS_REPO |  <--sensor_msgs::JointState YOUR_HARD_WARE_JOINTSTATE_TOPIC---- | HARDWARE_DRIVER |
    |           |  ---YOUR_CONTROL_INTERFACE_CLASS YOUR_CONTROL_INPUT_NAME -----> |                 |

and:

    link_names: ${YOUR_LINK_NAMES} # optional
    link_params: ${YOUR_LINK_PARAMS} # dynamical params, not optional
    dh_params: ${YOUR_DH_PARAMS} # optional

if you are using kinova kortex gen3 or franka panda, you can directly use the existing config files.

TODO: parse of config file to launch file (currenty manually)

### run (e.g. gen3):

    ## launch the kortex driver
    roslaunch kortex_driver kortex_arm_driver.launch ip_address:=YOUR_IP_ADDRESS
    ## launch the core nodes
    roslaunch speed_plan load_kortex_env_hw.launch
    ## launch the vison  nodes
    roslaunch human_interface task_e.launch ##  TODO: noptimizing the launch file
    ## launch the demo task node
    roslaunch speed_plan start_demo.launch

Under review paper: Fuzzy-Adaptive Compliance Control Method: Speed and Trajectory Optimization for Safe Human Robot Collaboration
