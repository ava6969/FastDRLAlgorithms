//
// Created by dewe on 3/31/21.
//

#ifndef CPPDRL_CARTPOLE_H
#define CPPDRL_CARTPOLE_H
#include <random>
#include <memory>
#include "torch/torch.h"
#include <unordered_map>
#include <cmath>
#include "gym_env.h"


using std::unique_ptr;
using std::vector;
using std::string;
using namespace torch;

const double PI = 3.141592653589793238463;

class CartPoleEnv : public GymEnv {

private:
    // GameRelated

    std::optional<int> stepsBeyondDone = std::nullopt;
    int maxEpisodeStep = 500;
    int rewardThreshold = 475.0;
    torch::Tensor state;

    /*shared_ptr<Viewer> viewer{};*/
    int stepCounter{};
    float gravity = 9.8;
    float masscart = 1.0;
    float masspole = 0.1;
    float total_mass = masspole + masscart;
    float length = 0.5;  // actually half the pole's length
    float polemass_length = masspole * length;
    float force_mag = 10.0;
    float tau = 0.02;  // seconds between state updates
    string kinematics_integrator = "euler";
    // Angle at which to fail episode
    double thetaThresholdRadians = 12 * 2 * PI / 360;
    double xThreshold = 2.4;


public:
    explicit CartPoleEnv(EnvConfig const &_env_config) : GymEnv(_env_config) {
        observation_space = std::make_unique<Box>(std::numeric_limits<double>::min(),
                                                  std::numeric_limits<double>::max(),
                                                  std::vector<int64_t>{4},
                                                  torch::kFloat32);

        action_space = std::make_unique<Discrete>(2);

    }

    StepResult step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const &action) override {

        auto x = state[0].unsqueeze(0);
        auto x_dot = state[1].unsqueeze(0);
        auto theta = state[2].unsqueeze(0);
        auto theta_dot = state[3].unsqueeze(0);
        double four_thirds = 4.0 / 3.0;

        float force = (get<torch::Tensor>(action).item<int>() == 1) ? force_mag : -force_mag;

        auto costheta = torch::cos(theta);
        auto sintheta = torch::sin(theta);

        auto temp = (force + polemass_length * theta_dot.pow(2) * sintheta) / total_mass;
        auto thetaacc = (gravity * sintheta - costheta * temp) /
                        (length * (four_thirds - masspole * costheta.pow(2) / total_mass));
        auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        if (kinematics_integrator == "euler") {
            x += tau * x_dot;
            x_dot += tau * xacc;
            theta += tau * theta_dot;
            theta_dot += tau * thetaacc;
        } else {
            x_dot += tau * xacc;
            x += tau * x_dot;
            theta_dot += tau * thetaacc;
            theta += tau * theta_dot;

        }


        state = torch::cat({x, x_dot, theta, theta_dot});

        auto done = x.item<float>() < -xThreshold ||
                    x.item<float>() > xThreshold ||
                    theta.item<float>() < -thetaThresholdRadians ||
                    theta.item<float>() > thetaThresholdRadians;

        ++stepCounter;
        done = done || stepCounter == maxEpisodeStep;
        float reward = 0;
        if (!done)
            reward = 1;

        else if (!stepsBeyondDone.has_value()) {

            // Pole just fell!
            stepsBeyondDone = 0;
            reward = 1.0;
        } else {
            if (stepsBeyondDone.value() == 0) {
                std::cout <<
                          "You are calling 'step()' even though this "
                          "environment has already returned done = True. You "
                          "should always call 'reset()' once you receive 'done = "
                          "True' -- any further steps are undefined behavior." << std::endl;
            }
            stepsBeyondDone = stepsBeyondDone.value() + 1;
            reward = 0.0;
        }

        return {state, torch::tensor(reward), torch::tensor(done), {}};
    }

    torch::Tensor reset() override {
        stepCounter = 0;
        stepsBeyondDone = std::nullopt;
        state = torch::rand(observation_space->getShape()).uniform_(-0.05, 0.05);
        return state;
    }

    std::optional<cv::Mat> render(bool human) override
    {
        return std::nullopt;
    }
};


#endif //CPPDRL_CARTPOLE_H
