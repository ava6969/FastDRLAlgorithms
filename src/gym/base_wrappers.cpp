//
// Created by dewe on 3/27/21.
//

#include "cppdrl/misc/helper.h"
#include "cppdrl/gym/base_wrappers.h"

//
//torch::Tensor Observation::reset()
//{
//    return preprocess(env->reset());
//}
//
//StepResult Observation::step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action)
//{
//    auto[next_obs, reward, done, info] = env->step(action);
//    return {preprocess(next_obs), reward, done, info};
//}
//
//void Observation::close()  {env->close(); };
