#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "indy7_msgs/msg/joint_state.hpp"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

namespace {
    struct CameraParams {
        static inline float azimuth = 90.0f;
        static inline float elevation = -20.0f;
        static inline float distance = 2.0f;
        static inline const float sensitivity = 0.5f;
    };

    struct MouseState {
        static inline bool drag = false;
        static inline double last_x = 0.0;
        static inline double last_y = 0.0;
    };

    void glfw_error_callback(int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
    }

    void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            MouseState::drag = (action == GLFW_PRESS);
        }
    }

    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        if (MouseState::drag) {
            CameraParams::azimuth -= (xpos - MouseState::last_x) * CameraParams::sensitivity;
            CameraParams::elevation -= (ypos - MouseState::last_y) * CameraParams::sensitivity;

            // Clamp elevation to prevent camera flipping
            CameraParams::elevation = std::clamp(CameraParams::elevation, -90.0f, 90.0f);
        }
        MouseState::last_x = xpos;
        MouseState::last_y = ypos;
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        CameraParams::distance *= 1.0f - yoffset * 0.1f;
    }
}

class MujocoSimNode : public rclcpp::Node {
public:
    MujocoSimNode() : Node("mujoco_sim_node") {
        initializeGLFW();
        initializeMujoco();
        initializeSimulation();
        initializeROSInterfaces();
    }

    ~MujocoSimNode() {
        cleanup();
    }

private:
    mjModel* model_{nullptr};
    mjData* data_{nullptr};
    std::mutex mutex_;
 
    mjvScene scene_;
    mjvCamera camera_;
    mjvOption options_;
    mjrContext context_;
    GLFWwindow* window_{nullptr};
    
    rclcpp::Publisher<indy7_msgs::msg::JointState>::SharedPtr state_pub_;
    rclcpp::Subscription<indy7_msgs::msg::JointState>::SharedPtr position_sub_;
    rclcpp::TimerBase::SharedPtr sim_timer_;
    
    double sim_time_{0.0};
    double sim_timestep_{0.01};  // 100Hz simulation TODO: move to config file
    rclcpp::Time current_sim_time_;
    rclcpp::Clock::SharedPtr sim_clock_;

    void initializeGLFW() {
        RCLCPP_INFO(get_logger(), "Initializing GLFW");
        
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        glfwSetErrorCallback(glfw_error_callback);

        window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulator", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_position_callback);
        glfwSetKeyCallback(window_, keyboard_callback);
        glfwSetScrollCallback(window_, scroll_callback);
    }

    void initializeMujoco() {
        RCLCPP_INFO(get_logger(), "Loading MuJoCo model");
        
        const std::string model_path = "indy7-control/src/mujoco_sim/models/indy7.xml";
        char error[1000];
        
        model_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!model_) {
            RCLCPP_ERROR(get_logger(), "Failed to load model: %s", error);
            throw std::runtime_error("Model loading failed");
        }

        // Disable gravity for testing purposes
        model_->opt.gravity[0] = 0;
        model_->opt.gravity[1] = 0;
        model_->opt.gravity[2] = 0;

        data_ = mj_makeData(model_);

        // Set initial joint positions to be near the start of figure 8 trajectory
        std::vector<double> initial_positions = {1.0, 0.0, -1.62, 0.0, -0.5, 0.0};
        for (size_t i = 0; i < initial_positions.size(); ++i) {
            data_->qpos[i] = initial_positions[i];
        }
        
        mj_forward(model_, data_);
    }

    void initializeSimulation() {
        mjv_defaultScene(&scene_);
        mjv_defaultCamera(&camera_);
        mjv_defaultOption(&options_);
        mjr_defaultContext(&context_);

        mjv_makeScene(model_, &scene_, 2000);
        mjr_makeContext(model_, &context_, mjFONTSCALE_150);

        camera_.type = mjCAMERA_FREE;
        camera_.distance = CameraParams::distance;
        camera_.azimuth = CameraParams::azimuth;
        camera_.elevation = CameraParams::elevation;

        // Configure position control, remove if we want to use torque control
        for (int i = 0; i < model_->nu; i++) {
            model_->actuator_gainprm[i * 3] = 100.0;     // Position gain
            model_->actuator_gainprm[i * 3 + 1] = 10.0;  // Velocity gain
        }

        model_->opt.timestep = sim_timestep_;

        sim_clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
        current_sim_time_ = sim_clock_->now();
    }

    void initializeROSInterfaces() {
        state_pub_ = create_publisher<indy7_msgs::msg::JointState>("joint_states", 10);
        position_sub_ = create_subscription<indy7_msgs::msg::JointState>(
            "joint_commands", 1,
            std::bind(&MujocoSimNode::commandCallback, this, std::placeholders::_1)
        );

        sim_timer_ = create_wall_timer(
            std::chrono::duration<double>(sim_timestep_),
            std::bind(&MujocoSimNode::simulationStep, this)
        );

        this->set_parameter(rclcpp::Parameter("use_sim_time", true));

        RCLCPP_INFO(get_logger(), "MuJoCo simulation node initialized with timestep: %f", sim_timestep_);
    }

    void cleanup() {
        if (data_) mj_deleteData(data_);
        if (model_) mj_deleteModel(model_);
        mjv_freeScene(&scene_);
        mjr_freeContext(&context_);
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void commandCallback(const indy7_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        RCLCPP_INFO(get_logger(), "Received position command: %f %f %f %f %f %f",
                    msg->positions[0], msg->positions[1], msg->positions[2],
                    msg->positions[3], msg->positions[4], msg->positions[5]);
        
        for (size_t i = 0; i < msg->positions.size(); ++i) {
            data_->ctrl[i] = msg->positions[i];
        }
    }

    void simulationStep() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        mj_step(model_, data_);
        sim_time_ += sim_timestep_;
        current_sim_time_ = current_sim_time_ + rclcpp::Duration::from_seconds(sim_timestep_);

        publishJointState();
        updateVisualization();
    }

    void publishJointState() {
        auto state_msg = std::make_unique<indy7_msgs::msg::JointState>();
        state_msg->header.stamp = current_sim_time_;
        
        for (int i = 0; i < model_->nv; ++i) {
            state_msg->positions[i] = data_->qpos[i];
            state_msg->velocities[i] = data_->qvel[i];
            state_msg->torques[i] = data_->ctrl[i];
        }

        state_pub_->publish(std::move(state_msg));
    }

    void updateVisualization() {
        camera_.azimuth = CameraParams::azimuth;
        camera_.elevation = CameraParams::elevation;
        camera_.distance = CameraParams::distance;

        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);

        mjrRect viewport = {0, 0, width, height};
        mjv_updateScene(model_, data_, &options_, nullptr, &camera_, mjCAT_ALL, &scene_);
        mjr_render(viewport, &scene_, &context_);

        glfwSwapBuffers(window_);
        glfwPollEvents();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MujocoSimNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
