#pragma once

#include "indydcp3.h"
#include "indy7_msgs/msg/joint_state.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <thread>

class Indy7JointController {
public:
    Indy7JointController(const std::string& robot_ip, int index = 0);
    ~Indy7JointController();

    void start_teleop();
    void stop_teleop();
    void move_joints_teleop(const std::vector<float>& q);
    void move_joints(const std::vector<float>& q);
    
    bool get_robot_state(indy7_msgs::msg::JointState& state);

private:
    IndyDCP3 indyDCP3;
};
