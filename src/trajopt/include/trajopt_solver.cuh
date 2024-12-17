#include <vector>
#include <string>
#include <cstring>
#include "dynamics/rbd_plant.cuh"
#include "gpu_pcg.cuh"
#include "pcg/sqp.cuh"

template <typename T, int StateSize=12, int ControlSize=6, int KnotPoints=128, int NumPcgThreads=128>
class TrajoptSolver {
public:
    TrajoptSolver(const std::vector<T>& goal_eePos_traj_1d,
                  const T timestep,
                  const T pcg_exit_tol,
                  const int pcg_max_iter) 
                  : timestep_(timestep),
                    pcg_exit_tol_(pcg_exit_tol),
                    pcg_max_iter_(pcg_max_iter),
                    xu_traj_length_((StateSize + ControlSize) * KnotPoints - ControlSize),
                    goal_traj_length_(goal_eePos_traj_1d.size() / 6),
                    last_update_time_(0) {
        // Initialize h_xu_ to zero
        std::memset(h_xu_, 0, xu_traj_length_ * sizeof(T));
        // PCG initialization
        checkPcgOccupancy<T>((void *) pcg<T, StateSize, KnotPoints>, NumPcgThreads, StateSize, KnotPoints);
        pcg_config_.pcg_block = NumPcgThreads;
        pcg_config_.pcg_exit_tol = pcg_exit_tol;
        pcg_config_.pcg_max_iter = pcg_max_iter;

        // --- Device memory allocation ---
        // Full goal trajectory
        gpuErrchk(cudaMalloc(&d_goal_eePos_traj_, goal_eePos_traj_1d.size() * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_goal_eePos_traj_, goal_eePos_traj_1d.data(), goal_eePos_traj_1d.size() * sizeof(T), cudaMemcpyHostToDevice));
        // End effector trajectory for trajopt (initialized with start of goal trajectory)
        gpuErrchk(cudaMalloc(&d_eePos_traj_, 6 * KnotPoints * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_eePos_traj_, d_goal_eePos_traj_, 6 * KnotPoints * sizeof(T), cudaMemcpyDeviceToDevice));
        // State trajectory
        gpuErrchk(cudaMalloc(&d_xu_, xu_traj_length_ * sizeof(T)));
        gpuErrchk(cudaMemset(d_xu_, 0, xu_traj_length_ * sizeof(T)));
        // Lagrange multipliers
        gpuErrchk(cudaMalloc(&d_lambda_, StateSize * KnotPoints * sizeof(T)));
        gpuErrchk(cudaMemset(d_lambda_, 0, StateSize * KnotPoints * sizeof(T)));
        // Robot model
        d_dynamics_const_mem_ = gato_plant::initializeDynamicsConstMem<T>();
    }

    ~TrajoptSolver() {
        // Synchronize and free device memory
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaFree(d_goal_eePos_traj_));
        gpuErrchk(cudaFree(d_eePos_traj_));
        gpuErrchk(cudaFree(d_xu_));
        gpuErrchk(cudaFree(d_lambda_));
        gato_plant::freeDynamicsConstMem<T>(d_dynamics_const_mem_);
    }

    void initializeXU(const std::vector<T>& current_joint_positions) {
        const int stride = StateSize + ControlSize;
        const int dofs = StateSize/2;
        for (int i = 0; i < KnotPoints - 1; i++) { // Iterate from first to penultimate knot point
            // Copy current position to xu vector
            std::memcpy(h_xu_ + i * stride, current_joint_positions.data(), dofs * sizeof(T));
            // Set velocity to 0
            std::memset(h_xu_ + i * stride + dofs, 0, dofs * sizeof(T));
            // Set torque to 0
            std::memset(h_xu_ + i * stride + 2*dofs, 0, ControlSize * sizeof(T));
        }
        // Set position and velocity of last knot point
        std::memcpy(h_xu_ + (KnotPoints - 1) * stride, current_joint_positions.data(), dofs * sizeof(T));
        std::memset(h_xu_ + (KnotPoints - 1) * stride + dofs, 0, dofs * sizeof(T));
        // Copy xu vector to device
        gpuErrchk(cudaMemcpy(d_xu_, h_xu_, xu_traj_length_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void warmStart() {
        pcg_config_.pcg_exit_tol = 1e-11;
        pcg_config_.pcg_max_iter = 10000;
        // Run solver a bunch of times to warm start lagrange multipliers
        for(int j = 0; j < 100; j++) {
            sqpSolvePcg<T>(StateSize, ControlSize, KnotPoints, timestep_, d_eePos_traj_, d_lambda_, d_xu_, d_dynamics_const_mem_, pcg_config_, rho_, 1e-3);
            // reset xu vector
            gpuErrchk(cudaMemcpy(d_xu_, h_xu_, xu_traj_length_ * sizeof(T), cudaMemcpyHostToDevice));
        }
        // Reset pcg configs
        rho_ = 1e-3;
        pcg_config_.pcg_exit_tol = pcg_exit_tol_;
        pcg_config_.pcg_max_iter = pcg_max_iter_;
    }

    std::string runTrajoptIteration() {
        const auto sqp_stats = sqpSolvePcg<T>(StateSize, ControlSize, KnotPoints, timestep_, d_eePos_traj_, d_lambda_, d_xu_, d_dynamics_const_mem_, pcg_config_, rho_, 1e-3);

        const auto& [pcg_iters, linsys_times, sqp_solve_time, sqp_iters, sqp_exit, pcg_exits] = sqp_stats;
        std::string status = sqp_exit ? "Success" : "Failure";
        return "SQP iterations: " + std::to_string(sqp_iters) + ", SQP solve time: " + std::to_string(sqp_solve_time) + " us, Exit status: " + status;
    }

    void shiftTrajectory(const std::vector<T>& current_state, const T current_time) { // current_state is of size StateSize
        // Calculate how many timesteps to shift based on elapsed time
        //TODO: need a more clever way to do this with start time, goal trajectory, and timestep
        // TODO: use interpolation instead of discrete timesteps

        // if last_update_time_ is 0, set it to current_time and return
        if (last_update_time_ == 0) {
            last_update_time_ = current_time;
            return;
        }

        T time_since_last_update = current_time - last_update_time_;
        int num_shifts = static_cast<int>(time_since_last_update / timestep_); 
        T time_since_last_shift = time_since_last_update - num_shifts * timestep_;

        // Add bounds checking for num_shifts to prevent overflow
        num_shifts = std::min(num_shifts, static_cast<int>(goal_traj_length_ - traj_offset_));
        
        if (num_shifts > 0) {
            // Update goal trajectory for all knot points
            if (traj_offset_ + KnotPoints <= goal_traj_length_) {
                // If we have enough points remaining, copy the next chunk of the goal trajectory
                gpuErrchk(cudaMemcpy(
                    d_eePos_traj_,
                    &d_goal_eePos_traj_[traj_offset_ * 6],
                    6 * KnotPoints * sizeof(T),
                    cudaMemcpyDeviceToDevice
                ));
            } else {
                // If we're near the end, copy remaining points and fill the rest with the final position
                int remaining_points = goal_traj_length_ - traj_offset_;
                if (remaining_points > 0) {
                    gpuErrchk(cudaMemcpy(
                        d_eePos_traj_,
                        &d_goal_eePos_traj_[traj_offset_ * 6],
                        6 * remaining_points * sizeof(T),
                        cudaMemcpyDeviceToDevice
                    ));
                    
                    // Fill remaining points with final position
                    for (int i = remaining_points; i < KnotPoints; i++) {
                        gpuErrchk(cudaMemcpy(
                            &d_eePos_traj_[i * 6],
                            &d_goal_eePos_traj_[(goal_traj_length_ - 1) * 6],
                            6 * sizeof(T),
                            cudaMemcpyDeviceToDevice
                        ));
                    }
                }
            }

            for (int i = 0; i < num_shifts; i++) {
                if (traj_offset_ >= goal_traj_length_) { break; }
                traj_offset_++;
                
                // Shift xu and lambda trajectories
                just_shift<T>(StateSize, ControlSize, KnotPoints, d_xu_);
                just_shift<T>(StateSize, 0, KnotPoints, d_lambda_);

                // Fill in x_N = x_N-1, set u_N-1 to 0
                gpuErrchk(cudaMemcpy(
                    &d_xu_[xu_traj_length_ - StateSize], 
                    &d_xu_[xu_traj_length_-(2*StateSize)-ControlSize], 
                    StateSize * sizeof(T), 
                    cudaMemcpyDeviceToDevice
                ));
                gpuErrchk(cudaMemset(
                    &d_xu_[xu_traj_length_ - (StateSize + ControlSize)], 
                    0, 
                    ControlSize * sizeof(T)
                ));
            }
            
            last_update_time_ = current_time - time_since_last_shift;
        }

        // Update x_0 with current state
        gpuErrchk(cudaMemcpy(d_xu_, current_state.data(), StateSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    bool isTrajectoryComplete() const {
        return traj_offset_ >= goal_traj_length_;
    }

    uint32_t getTrajectoryOffset() const {
        return traj_offset_;
    }

    uint32_t numKnotPoints() const {
        return KnotPoints;
    }

    uint32_t stateSize() const {
        return StateSize;
    }

    uint32_t controlSize() const {
        return ControlSize;
    }

    std::pair<const T*, size_t> getOptimizedTrajectory() {
        // Copy current trajectory from device to host
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_xu_, d_xu_, xu_traj_length_ * sizeof(T), cudaMemcpyDeviceToHost));
        return {h_xu_, xu_traj_length_};
    }

private:
    pcg_config<T> pcg_config_;
    
    T h_xu_[StateSize * KnotPoints + ControlSize * (KnotPoints - 1)];
    const int xu_traj_length_;

    T *d_goal_eePos_traj_;
    T *d_eePos_traj_;
    T *d_xu_;
    T *d_lambda_;
    void *d_dynamics_const_mem_;
    
    const T timestep_;
    const T pcg_exit_tol_;
    const int pcg_max_iter_;
    uint32_t traj_offset_{0};  // Current offset in the goal trajectory
    const uint32_t goal_traj_length_;  // Total length of goal trajectory
    T rho_{1e-3};
    
    std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>> sqp_stats;

    T last_update_time_;
};
