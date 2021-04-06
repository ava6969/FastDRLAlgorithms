//
// Created by dewe on 3/27/21.
//

#pragma once

#include "variant"
#include "map"
#include "string"
#include <memory>
#include <torch/torch.h>
#include "spaces/space.h"
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <cppdrl/misc/helper.h>

using std::unique_ptr;
struct StepResult;
using std::string;
using std::map;

class GymEnv{

protected:
    unique_ptr<Space> observation_space{};
    unique_ptr<Space> action_space{};
    EnvConfig env_config;

public:

    GymEnv( EnvConfig const& _env_config):env_config(_env_config) {}
    virtual ~GymEnv()=default;
    virtual torch::Tensor reset() = 0;
    virtual StepResult step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action) = 0;

    [[nodiscard]] Space* ObservationSpace() const { return observation_space.get();}
    [[nodiscard]] Space* Actionspace() const  { return action_space.get();}
    virtual void close() {};

    virtual std::optional<cv::Mat> render(bool human) = 0;
//
//    friend class AtariPreprocessing; // temporary

};

