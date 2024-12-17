#include "joint_controller.hpp"

Indy7JointController::Indy7JointController(const std::string& robot_ip, int index)
    : indyDCP3(robot_ip, index) {
    start_teleop();
    std::this_thread::sleep_for(std::chrono::seconds(1)); // wait for the robot to be ready
}

Indy7JointController::~Indy7JointController() {
    stop_teleop();
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void Indy7JointController::start_teleop() {
    bool success = indyDCP3.start_teleop(TeleMethod::TELE_JOINT_ABSOLUTE);
    if (!success) {
        std::cerr << "Failed to start teleoperation." << std::endl;
    }
}

void Indy7JointController::stop_teleop() {
    bool success = indyDCP3.stop_teleop();
    if (success) {
        std::cout << "Teleoperation stopped successfully." << std::endl;
    } else {
        std::cerr << "Failed to stop teleoperation." << std::endl;
    }
}

void Indy7JointController::move_joints_teleop(const std::vector<float>& q) {
    bool success = indyDCP3.movetelej(q, 1.0, 1.0, TeleMethod::TELE_JOINT_ABSOLUTE);
    if (!success) {
        std::cerr << "Failed to move joint positions in teleoperation mode." << std::endl;
    }
}

void Indy7JointController::move_joints(const std::vector<float>& q) {
    bool success = indyDCP3.movej(q, JointBaseType::ABSOLUTE_JOINT, BlendingType_Type::BlendingType_Type_NONE);
    if (!success) {
        std::cerr << "MoveJ command failed." << std::endl;
    }
}

bool Indy7JointController::get_robot_state(indy7_msgs::msg::JointState& state) {
    Nrmk::IndyFramework::ControlData control_state;
    bool success = indyDCP3.get_robot_data(control_state);
    
    if (!success) {
        std::cerr << "Failed to get control state." << std::endl;
        return false;
    }

    // Copy data from control_state to JointState message
    for (int i = 0; i < 6; i++) {
        state.positions[i] = control_state.q(i);
        state.velocities[i] = control_state.qdot(i);
    }

    return true;
}

