#include <filesystem>
#include <chrono>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "indy7_msgs/msg/joint_state.hpp"
#include "indy7_msgs/msg/joint_trajectory.hpp"
#include "indy7_msgs/msg/joint_trajectory_point.hpp"
#include "trajopt_solver.cuh"
#include "csv_utils.h"

namespace msgs = indy7_msgs::msg;
using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;
using SimTime = rclcpp::Time;

class TrajoptNode : public rclcpp::Node
{
public:
    explicit TrajoptNode(const std::string& traj_file)
        : Node("trajopt_node")
        , timestep_(Duration(0.01))
        , pcg_exit_tol_(5e-4)
        , pcg_max_iter_(173)
        , state_updated_(false)
        , warm_start_complete_(false)
        , optimization_in_progress_(false)
        , use_sim_time_(false)
    {
        use_sim_time_ = get_parameter("use_sim_time").as_bool();
        RCLCPP_INFO(this->get_logger(), "Using %s time", use_sim_time_ ? "simulation" : "system");
        RCLCPP_INFO(this->get_logger(), "Initializing TrajoptNode");

        setupCommunication();
        initializeSolver(traj_file);
        waitForInitialState();
    }

    ~TrajoptNode() {
        RCLCPP_INFO(this->get_logger(), "Shutting down TrajoptNode");
    }

private:
    void setupCommunication() {
        state_sub_ = create_subscription<msgs::JointState>(
            "joint_states", 1, 
            std::bind(&TrajoptNode::stateCallback, this, std::placeholders::_1)
        );
        traj_pub_ = create_publisher<msgs::JointTrajectory>("joint_trajectory", 1);
    }

    void initializeSolver(const std::string& traj_file) {
        std::vector<float> goal_eePos_traj_1d = readCsvToVector<float>(traj_file);
        solver_ = std::make_unique<TrajoptSolver<float>>(
            goal_eePos_traj_1d,
            timestep_.count(),
            pcg_exit_tol_,
            pcg_max_iter_
        );
        RCLCPP_INFO(this->get_logger(), "Solver initialized");

        // Pre-allocate trajectory message
        traj_msg_.knot_points = solver_->numKnotPoints();
        traj_msg_.points.resize(solver_->numKnotPoints());
        full_state_.reserve(solver_->stateSize());
    }

    void waitForInitialState() {
        RCLCPP_INFO(this->get_logger(), "Waiting for initial state...");
        while (rclcpp::ok() && !state_updated_) {
            rclcpp::spin_some(this->get_node_base_interface());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::vector<float> current_joint_positions(
            current_state_.positions.begin(), 
            current_state_.positions.end()
        );
        solver_->initializeXU(current_joint_positions);
        
        RCLCPP_INFO(this->get_logger(), "Received initial state, starting solver warm start...");
        solver_->warmStart();
        cudaDeviceSynchronize();
        warm_start_complete_ = true;
        RCLCPP_INFO(this->get_logger(), "Warm start complete");
        initializeStartTime();
    }

    void initializeStartTime() {
        if (use_sim_time_) {
            while (rclcpp::ok() && current_state_.header.stamp.sec == 0) {
                rclcpp::spin_some(this->get_node_base_interface());
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            trajectory_start_stamp_ = current_state_.header.stamp;
        } else {
            trajectory_start_time_ = Clock::now();
        }
    }

    void stateCallback(const std::shared_ptr<const msgs::JointState>& msg) {
        if (optimization_in_progress_.load() || solver_->isTrajectoryComplete()) {
            return;
        }

        updateCurrentState(msg);
        if (!warm_start_complete_) { return; }
        optimization_in_progress_ = true;
        runOptimization(msg);
        optimization_in_progress_ = false;
    }

    void updateCurrentState(const std::shared_ptr<const msgs::JointState>& msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_ = *msg;
        state_updated_ = true;

        full_state_.clear();
        full_state_.insert(full_state_.end(), 
            current_state_.positions.begin(), 
            current_state_.positions.end());
        full_state_.insert(full_state_.end(), 
            current_state_.velocities.begin(), 
            current_state_.velocities.end());
    }

    void runOptimization(const std::shared_ptr<const msgs::JointState>& msg) {
        solver_->shiftTrajectory(full_state_, calculateElapsedTime(msg->header.stamp));
        
        std::string stats = solver_->runTrajoptIteration();
        RCLCPP_INFO(this->get_logger(), "Optimization stats: %s", stats.c_str());
        RCLCPP_INFO(this->get_logger(), "Trajectory offset: %u", solver_->getTrajectoryOffset());

        publishTrajectory(msg->header.stamp);
    }

    double calculateElapsedTime(const builtin_interfaces::msg::Time& current_stamp) const {
        if (use_sim_time_) {
            SimTime current_time(current_stamp);
            SimTime start_time(trajectory_start_stamp_);
            return (current_time - start_time).seconds();
        }
        return Duration(Clock::now() - trajectory_start_time_).count();
    }

    void publishTrajectory(const builtin_interfaces::msg::Time& stamp) {
        auto traj_msg = msgs::JointTrajectory();
        traj_msg.header.stamp = stamp;
        traj_msg.knot_points = solver_->numKnotPoints();
        traj_msg.dt = timestep_.count();

        const auto [traj_data, traj_size] = solver_->getOptimizedTrajectory();
        const int stride = solver_->stateSize() + solver_->controlSize();
        
        traj_msg.points.clear();
        for (int i = 0; i < traj_msg.knot_points; i++) {
            msgs::JointTrajectoryPoint point;
            for (size_t j = 0; j < 6; ++j) {
                point.positions[j] = traj_data[i * stride + j];
                point.velocities[j] = traj_data[i * stride + solver_->stateSize()/2 + j];
                point.torques[j] = traj_data[i * stride + solver_->stateSize() + j];
            }
            traj_msg.points.emplace_back(point);
        }

        traj_pub_->publish(traj_msg);
    }

    // ROS communication
    rclcpp::Subscription<msgs::JointState>::SharedPtr state_sub_;
    rclcpp::Publisher<msgs::JointTrajectory>::SharedPtr traj_pub_;

    // Solver configuration
    std::unique_ptr<TrajoptSolver<float, 12, 6, 128, 128>> solver_;
    const Duration timestep_;
    const float pcg_exit_tol_;
    const int pcg_max_iter_;

    // State management
    msgs::JointState current_state_;
    std::vector<float> full_state_;
    std::atomic<bool> state_updated_;
    std::mutex state_mutex_;

    // Trajectory management
    msgs::JointTrajectory traj_msg_;
    TimePoint trajectory_start_time_;
    builtin_interfaces::msg::Time trajectory_start_stamp_;

    // Control flags
    std::atomic<bool> warm_start_complete_;
    std::atomic<bool> optimization_in_progress_;
    bool use_sim_time_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajoptNode>(argv[1]);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}