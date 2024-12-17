#include "rclcpp/rclcpp.hpp"
#include <memory>
#include <mutex>
#include <chrono>
#include "indy7_msgs/msg/joint_trajectory.hpp"
#include "indy7_msgs/msg/joint_state.hpp"
#include "indy7_msgs/msg/joint_command.hpp"
#include "builtin_interfaces/msg/time.hpp"

namespace msgs = indy7_msgs::msg;
using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

class SimDriverNode : public rclcpp::Node {
public:
    SimDriverNode()
    : Node("sim_driver_node"),
      executing_trajectory_(false),
      current_point_index_(0),
      timestep_(std::chrono::duration<double>(0.01)) // TODO: make this a launch argument
    {
        initializeBuffers();
        setupSubscriptionsAndPublishers();
        RCLCPP_INFO(this->get_logger(), "Initializing SimDriverNode");
    }

private:
    void initializeBuffers() {
        state_msg_ = std::make_unique<msgs::JointState>();
        position_buffer_.resize(6, 0.0f);
        velocity_buffer_.resize(6, 0.0f);
    }

    void setupSubscriptionsAndPublishers() {
        trajectory_sub_ = create_subscription<msgs::JointTrajectory>(
            "joint_trajectory", 1,
            std::bind(&SimDriverNode::trajectoryCallback, this, std::placeholders::_1)
        );

        state_sub_ = create_subscription<msgs::JointState>(
            "joint_states", 1,
            std::bind(&SimDriverNode::stateCallback, this, std::placeholders::_1)
        );

        command_pub_ = create_publisher<msgs::JointState>("joint_commands", 1);

        trajectory_timer_ = create_wall_timer(
            std::chrono::milliseconds(1), // 1kHz
            std::bind(&SimDriverNode::trajectoryExecutionCallback, this)
        );
    }

    void trajectoryCallback(const msgs::JointTrajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_trajectory_ = std::move(*msg);
        current_point_index_ = 0;
        executing_trajectory_ = true;
        trajectory_start_time_ = msg->header.stamp; 
        RCLCPP_INFO(this->get_logger(), "Received new trajectory with %zu points", current_trajectory_.points.size());
    }

    void stateCallback(const msgs::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_ = *msg;

        if (executing_trajectory_) {
            const rclcpp::Time current_time(msg->header.stamp);
            const rclcpp::Time start_time(trajectory_start_time_);
            const double elapsed_time = (current_time - start_time).seconds();
            
            const size_t target_index = static_cast<size_t>(elapsed_time / timestep_.count());
            
            if (target_index >= current_trajectory_.points.size()) {
                executing_trajectory_ = false;
                RCLCPP_INFO(this->get_logger(), "Trajectory execution complete");
                return;
            }

            auto command_msg = std::make_unique<msgs::JointState>();
            command_msg->header.stamp = this->now();
            
            const auto& target = current_trajectory_.points[target_index];
            {
                std::lock_guard<std::mutex> state_lock(state_mutex_);
                for (size_t i = 0; i < 6; ++i) {
                    command_msg->positions[i] = target.positions[i];
                    command_msg->velocities[i] = target.velocities[i];
                    command_msg->torques[i] = target.torques[i];
                }
            }

            command_pub_->publish(std::move(command_msg));
        }
    }

    void trajectoryExecutionCallback() {
        if (!executing_trajectory_) {
            return;
        }

        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        
        const auto current_time = Clock::now();
        auto trajectory_start_timepoint = 
            std::chrono::system_clock::time_point(
                std::chrono::seconds(trajectory_start_time_.sec) + 
                std::chrono::nanoseconds(trajectory_start_time_.nanosec)
            );
        
        const Duration elapsed_time = current_time - trajectory_start_timepoint;
        const size_t target_index = static_cast<size_t>(elapsed_time.count() / timestep_.count());
        
        if (target_index >= current_trajectory_.points.size()) {
            executing_trajectory_ = false;
            RCLCPP_INFO(this->get_logger(), "Trajectory execution complete");
            return;
        }

        auto command_msg = std::make_unique<msgs::JointState>();
        command_msg->header.stamp = this->now();
        
        const auto& target = current_trajectory_.points[target_index];
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            for (size_t i = 0; i < 6; ++i) {
                command_msg->positions[i] = target.positions[i];
                command_msg->velocities[i] = target.velocities[i];
                command_msg->torques[i] = target.torques[i];
            }
        }

        command_pub_->publish(std::move(command_msg));
    }

    rclcpp::Subscription<msgs::JointTrajectory>::SharedPtr trajectory_sub_;
    rclcpp::Subscription<msgs::JointState>::SharedPtr state_sub_;
    rclcpp::Publisher<msgs::JointState>::SharedPtr command_pub_;
    rclcpp::TimerBase::SharedPtr trajectory_timer_;

    msgs::JointTrajectory current_trajectory_;
    msgs::JointState current_state_;
    size_t current_point_index_;
    bool executing_trajectory_;
    builtin_interfaces::msg::Time trajectory_start_time_;
    Duration timestep_;

    std::mutex trajectory_mutex_;
    std::mutex state_mutex_;

    std::unique_ptr<msgs::JointState> state_msg_;
    std::vector<float> position_buffer_;
    std::vector<float> velocity_buffer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimDriverNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 