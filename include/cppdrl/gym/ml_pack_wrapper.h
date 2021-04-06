//
// Created by dewe on 3/30/21.
//

#ifndef CPPDRL_ML_PACK_WRAPPER_H
#define CPPDRL_ML_PACK_WRAPPER_H

#include "cppdrl/gym/gym_env.h"
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include "memory"
#include "numeric"
#include "spaces/box.h"
#include "spaces/discrete.h"


using std::make_unique;

class CartPoleGym: public GymEnv
{
private:

    mlpack::rl::CartPole::State last_state;
    mlpack::rl::CartPole env;
public:
    explicit CartPoleGym(EnvConfig const& env_config):GymEnv(env_config)
    {
        env = mlpack::rl::CartPole();
        observation_space = std::make_unique<Box>(std::numeric_limits<double>::min(),
                                             std::numeric_limits<double>::max(),
                                             std::vector<int64_t>{4},
                                             torch::kFloat32);

        action_space = std::make_unique<Discrete>(2);
    }

    torch::Tensor reset() override
    {
        last_state = env.InitialSample();
        auto tensor = torch::from_blob(last_state.Data().memptr(), {4}, torch::kF64).toType(torch::kF32);
        return tensor;
    }
    StepResult step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action) override
    {
        using Action = mlpack::rl::CartPole::Action;
        Action::actions action_ = std::get<torch::Tensor>(action).item<int>() == 0 ? Action::actions::backward : Action::actions::forward;
        mlpack::rl::CartPole::State next_state;

        auto reward = env.Sample(last_state, Action{action_}, next_state);
        last_state = next_state;
        auto obs = torch::from_blob(last_state.Data().memptr(), {4}, torch::kF64).toType(torch::kF32);
        return {obs,
                torch::tensor(reward),
                torch::tensor(env.IsTerminal(last_state)),
                {}};
    }

    virtual void close() {};

    virtual std::optional<cv::Mat> render(bool human) { return std::nullopt; }
};

#endif //CPPDRL_ML_PACK_WRAPPER_H
