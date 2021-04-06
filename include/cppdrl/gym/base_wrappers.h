//
// Created by dewe on 3/27/21.
//

#pragma once

#include "gym_env.h"


//
//class Wrapper: public GymEnv
//{
//protected:
//    unique_ptr<GymEnv> env;
//public:
//    explicit Wrapper(unique_ptr<GymEnv> env):GymEnv(), env(std::move(env)) {}
//};
//


//class Observation: public Wrapper
//{
//
//public:
//    explicit Observation(GymEnv* env):Wrapper(env) {}
//
//    virtual torch::Tensor preprocess(torch::Tensor const& observation) = 0;
//
//    torch::Tensor reset() override;
//
//    StepResult step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action) override;
//
//    void close() override;
//
//};

