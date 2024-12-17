#include "rclcpp/rclcpp.hpp"
#include <memory>
#include <mutex>
#include <chrono>
#include "indydcp3.h"
#include "indy7_msgs/msg/joint_trajectory.hpp"
#include "indy7_msgs/msg/joint_state.hpp"
#include "joint_controller.hpp"

namespace msgs = indy7_msgs::msg;
using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

class RobotDriverNode : public rclcpp::Node {
public:
    RobotDriverNode(const std::string& robot_ip)
    : Node("robot_driver_node"),
      controller_(robot_ip),
      executing_trajectory_(false),
      state_msg_(std::make_unique<msgs::JointState>()),
      position_buffer_(6, 0.0f),
      timestep_(std::chrono::duration<double>(0.01)),
      trajectory_start_time_()
    {
        RCLCPP_INFO(this->get_logger(), "Initializing RobotDriverNode");

        trajectory_sub_ = create_subscription<msgs::JointTrajectory>(
            "joint_trajectory", 1,
            std::bind(&RobotDriverNode::trajectoryCallback, this, std::placeholders::_1)
        );

        state_pub_ = create_publisher<msgs::JointState>("joint_states", 1);

        state_timer_ = create_wall_timer(
            std::chrono::milliseconds(1),  // 1kHz state publishing
            std::bind(&RobotDriverNode::statePublishCallback, this)
        );

        trajectory_timer_ = create_wall_timer(
            std::chrono::milliseconds(1),  // 1kHz for trajectory execution
            std::bind(&RobotDriverNode::trajectoryExecutionCallback, this)
        );
    }

    ~RobotDriverNode() {
        RCLCPP_INFO(this->get_logger(), "Shutting down RobotDriverNode");
    }

private:
    void trajectoryCallback(const msgs::JointTrajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_trajectory_.points.reserve(msg->points.size());
        current_trajectory_ = std::move(*msg);
        current_point_index_ = 0;
        executing_trajectory_ = true;
        trajectory_start_time_ = Clock::now();
    }

    void statePublishCallback() {
        state_msg_->header.stamp = this->now();
        if (!controller_.get_robot_state(*state_msg_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get robot state");
            return;
        }
        state_pub_->publish(*state_msg_);
    }

    void trajectoryExecutionCallback() {
        if (!executing_trajectory_) {
            return;
        }

        const std::lock_guard<std::mutex> lock(trajectory_mutex_);
        
        const auto current_time = Clock::now();
        const Duration elapsed_time = current_time - trajectory_start_time_;
        const size_t target_index = static_cast<size_t>(elapsed_time.count() / timestep_.count());
        
        // Check if we've reached the end of the trajectory
        if (target_index >= current_trajectory_.points.size()) {
            executing_trajectory_ = false;
            RCLCPP_INFO(this->get_logger(), "Trajectory execution complete");
            return;
        }

        // Execute the trajectory point
        const auto& point = current_trajectory_.points[target_index];
        for (size_t i = 0; i < point.positions.size(); ++i) {
            position_buffer_[i] = point.positions[i];
        }
        controller_.move_joints_teleop(position_buffer_);
    }

    Indy7JointController controller_;
    rclcpp::Subscription<msgs::JointTrajectory>::SharedPtr trajectory_sub_;
    rclcpp::Publisher<msgs::JointState>::SharedPtr state_pub_;
    rclcpp::TimerBase::SharedPtr state_timer_;
    rclcpp::TimerBase::SharedPtr trajectory_timer_;

    msgs::JointTrajectory current_trajectory_;
    size_t current_point_index_;
    bool executing_trajectory_;
    TimePoint trajectory_start_time_;
    std::mutex trajectory_mutex_;
    Duration timestep_;
    
    std::unique_ptr<msgs::JointState> state_msg_;
    std::vector<float> position_buffer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<RobotDriverNode>("160.39.102.52"); // TODO: make this a launch argument
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


