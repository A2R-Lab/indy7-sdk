#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include <stdexcept>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include <GLFW/glfw3.h>

// Forward declaration of MujocoSimNode class
class MujocoSimNode;

namespace {
    struct CameraParams {
        static inline float azimuth = 60.0f;
        static inline float elevation = -45.0f;
        static inline float distance = 4.0f;
        static inline const float sensitivity = 0.5f;
    };

    struct MouseState {
        static inline bool drag = false;
        static inline double last_x = 0.0;
        static inline double last_y = 0.0;
    };

    // Flag to indicate that a reset is requested
    static inline bool reset_requested = false;

#ifdef ENABLE_VISUALIZATION
    void glfw_error_callback(int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
    }

    void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        
        // Set reset flag when 'R' key is pressed
        if (key == GLFW_KEY_R && action == GLFW_PRESS) {
            reset_requested = true;
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

#else
    void initializeGLFW() {
        return;
        //RCLCPP_INFO(get_logger(), "Visualization disabled");
    }   
#endif
}

// Global pointer to the MujocoSimNode instance
MujocoSimNode* g_mujoco_sim_node = nullptr;

class MujocoSimNode : public rclcpp::Node {
public:
    MujocoSimNode(std::string model_path, double sim_timestep, bool enable_visualization) 
        : Node("mujoco_sim_node"), 
          sim_timestep_(sim_timestep),
          enable_visualization_(enable_visualization) {
        
        // Set the global pointer to this instance
        g_mujoco_sim_node = this;
        
        if (enable_visualization_) {
            initializeGLFW();
        }
        
        initializeMujoco(model_path);
        initializeSimulation();
        initializeROSInterfaces();
    }

    ~MujocoSimNode() {
        // Clear the global pointer
        g_mujoco_sim_node = nullptr;
        cleanup();
    }

    // Add a public method to reset the robot state
    void resetRobotState() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Reset to initial positions
        std::vector<double> initial_positions = {1.57994932,  0.06313131, -1.18071498,  1.09272369, -0.6255293,  -0.01895007};
        for (size_t i = 0; i < initial_positions.size() && i < model_->nv; ++i) {
            data_->qpos[i] = initial_positions[i];
            data_->qvel[i] = 0.0;
        }
        
        // Reset controls
        for (int i = 0; i < model_->nu; ++i) {
            data_->ctrl[i] = 0.0;
        }
        
        // Reset command received flag to wait for a new command
        command_received_ = false;
        
        // Forward the model to update the state
        mj_forward(model_, data_);
        
        RCLCPP_INFO(get_logger(), "Robot state reset to initial position. Waiting for new command.");
    }

private:
    mjModel* model_{nullptr};
    mjData* data_{nullptr};
    std::mutex mutex_;
 
    mjvScene scene_;
    mjvCamera camera_;
    mjvOption options_;
    mjrContext context_;
    
#ifdef ENABLE_VISUALIZATION
    GLFWwindow* window_{nullptr};
#endif
    
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr position_sub_;
    rclcpp::TimerBase::SharedPtr sim_timer_;
    
    double sim_timestep_;
    double total_sim_time_{0.0};
    rclcpp::Time current_sim_time_;
    rclcpp::Clock::SharedPtr sim_clock_;
    
    std::vector<std::string> joint_names_{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    
    // External force parameters
    bool apply_external_force_{false};
    std::vector<double> external_force_{0.0, 0.0, 0.0};
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr external_force_sub_;
    
    // Visualization flag
    bool enable_visualization_;
    
    // Flag to track if a command has been received
    bool command_received_{false};

#ifdef ENABLE_VISUALIZATION
    void initializeGLFW() {
        RCLCPP_INFO(get_logger(), "Initializing GLFW");
        if (!glfwInit()) { throw std::runtime_error("Failed to initialize GLFW"); }
        glfwSetErrorCallback(glfw_error_callback);
        window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulator", nullptr, nullptr);
        if (!window_) { glfwTerminate(); throw std::runtime_error("Failed to create GLFW window"); }
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_position_callback);
        glfwSetKeyCallback(window_, keyboard_callback);
        glfwSetScrollCallback(window_, scroll_callback);
    }
#endif

    void initializeMujoco(std::string model_path) {
        RCLCPP_INFO(get_logger(), "loading MuJoCo model from %s", model_path.c_str());
        char error[1000];
        model_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!model_) {
            RCLCPP_ERROR(get_logger(), "Failed to load model: %s", error);
            throw std::runtime_error("Model loading failed");
        }
        // disable gravity
        // model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2] = 0, 0, 0;
        data_ = mj_makeData(model_);

        std::vector<double> initial_positions = {1.57994932,  0.06313131, -1.18071498,  1.09272369, -0.6255293,  -0.01895007}; // TODO: make a parameter
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
        
        if (enable_visualization_) {
            mjv_makeScene(model_, &scene_, 2000);
            mjr_makeContext(model_, &context_, mjFONTSCALE_150);
            camera_.type = mjCAMERA_FREE;
            camera_.distance = CameraParams::distance;
            camera_.azimuth = CameraParams::azimuth;
            camera_.elevation = CameraParams::elevation;
        }
        
        model_->opt.timestep = sim_timestep_;
        sim_clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
        current_sim_time_ = sim_clock_->now();
    }

    void initializeROSInterfaces() {
        // joint states publisher
        state_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);

        // torques from controller
        position_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "joint_commands", 1,
            std::bind(&MujocoSimNode::commandCallback, this, std::placeholders::_1)
        );
        
        // External force subscription
        external_force_sub_ = create_subscription<geometry_msgs::msg::Wrench>(
            "external_force", 1,
            std::bind(&MujocoSimNode::externalForceCallback, this, std::placeholders::_1)
        );

        // simulation steps
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
        
        if (enable_visualization_) {
            mjv_freeScene(&scene_);
            mjr_freeContext(&context_);
#ifdef ENABLE_VISUALIZATION
            if (window_) glfwDestroyWindow(window_);
            glfwTerminate();
#endif
        }
    }

    void commandCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (msg->position.size() < 6) {
            RCLCPP_WARN(get_logger(), "Received incomplete joint command, expected 6 joint angles");
            return;
        }

        for (size_t i = 0; i < 6 && i < msg->effort.size(); ++i) {
            data_->ctrl[i] = msg->effort[i];
        }

        // RCLCPP_INFO(get_logger(), "Received controls: %.4f %.4f %.4f %.4f %.4f %.4f",
        //             msg->effort[0], msg->effort[1], msg->effort[2], msg->effort[3], msg->effort[4], msg->effort[5]);
        // ---
        // Set the command received flag
        command_received_ = true;
    }
    

    void simulationStep() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if reset was requested
        if (reset_requested) {
            resetRobotState();
            reset_requested = false;
        }
        
        // Only step the simulation if a command has been received
        if (command_received_) {
            mj_step(model_, data_);
        } else {
            // If no command received yet, just forward the model to update visualization
            mj_forward(model_, data_);
        }
        
        total_sim_time_ += sim_timestep_;
        publishJointState();
        
        if (enable_visualization_) {
            updateVisualization();
        }
    }

    void applyExternalForce() {
        // Find the end effector body ID (assuming it's the last body in the model)
        int end_effector_id = model_->nbody - 1;
        
        // Get the end effector position in world frame
        mjtNum pos[3];
        mju_copy3(pos, data_->xpos + 3 * end_effector_id);
        
        // Apply force in world frame
        mjtNum force[3] = {external_force_[0], external_force_[1], external_force_[2]};
        mjtNum torque[3] = {0, 0, 0}; // No torque applied
        
        // Apply the force and torque to the end effector
        mj_applyFT(model_, data_, force, torque, pos, end_effector_id, data_->qfrc_applied);

        RCLCPP_INFO(rclcpp::get_logger("mujoco_sim_node"), "applied force: %.4f %.4f %.4f on end effector", force[0], force[1], force[2]);
    }

    void publishJointState() {
        auto state_msg = std::make_unique<sensor_msgs::msg::JointState>();
        current_sim_time_ = current_sim_time_ + rclcpp::Duration::from_seconds(sim_timestep_);
        state_msg->header.stamp = current_sim_time_;
        state_msg->name = joint_names_;

        // TODO: needed?
        state_msg->position.resize(model_->nv);
        state_msg->velocity.resize(model_->nv);
        state_msg->effort.resize(model_->nv);
        
        for (int i = 0; i < model_->nv; ++i) {
            state_msg->position[i] = data_->qpos[i];
            state_msg->velocity[i] = data_->qvel[i];
            state_msg->effort[i] = data_->qfrc_applied[i];
        }
        for (int i = 0; i < 3; ++i) {
            state_msg->effort[i] = data_->xpos[3 * (model_->nbody - 1) + i];
        }
        
        state_pub_->publish(std::move(state_msg));
    }

#ifdef ENABLE_VISUALIZATION
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
#else
    void updateVisualization() {
        //RCLCPP_INFO(get_logger(), "Visualization disabled");
        return;
    }
#endif

    void externalForceCallback(const geometry_msgs::msg::Wrench::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        external_force_[0] = msg->force.x;
        external_force_[1] = msg->force.y;
        external_force_[2] = msg->force.z;
        apply_external_force_ = true;
        
        RCLCPP_INFO(rclcpp::get_logger("mujoco_sim_node"), "Received external force: [%f, %f, %f]",
                    external_force_[0], external_force_[1], external_force_[2]);

        applyExternalForce();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    // Parse command line arguments
    if (argc < 3) {
        RCLCPP_ERROR(rclcpp::get_logger("mujoco_sim_node"), 
                     "Usage: %s <model_path> <sim_timestep> [enable_visualization]", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    double sim_timestep = std::stod(argv[2]);
    bool enable_visualization = true; // Default to visualization enabled
    
    // Check if visualization flag is provided
    if (argc > 3) {
        std::string vis_flag = argv[3];
        enable_visualization = (vis_flag == "true" || vis_flag == "1");
    }
    
    RCLCPP_INFO(rclcpp::get_logger("mujoco_sim_node"), 
                "Starting MuJoCo simulation with visualization: %s", 
                enable_visualization ? "enabled" : "disabled");
    
    auto node = std::make_shared<MujocoSimNode>(model_path, sim_timestep, enable_visualization);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
