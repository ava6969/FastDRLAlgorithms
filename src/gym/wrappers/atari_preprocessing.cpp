//
// Created by dewe on 3/27/21.
//

#include "cppdrl/misc/helper.h"
#include "cppdrl/gym/wrappers/atari_preprocessing.h"
#include "cppdrl/gym/base_atari.h"
#include "cppdrl/gym/spaces/space.h"
#include "cppdrl/gym/spaces/box.h"
#include "cppdrl/gym/spaces/discrete.h"
#include "iostream"

using std::cout;
using std::endl;

AtariPreprocessing::AtariPreprocessing(unique_ptr<BaseAtari> _env,
                   const int noop_max,
                   const int frame_skip,
                   const int screen_size,
                   const bool terminal_on_life_loss,
                   const bool grayscale_obs,
                   const bool grayscale_new_axis,
                   const bool scale_obs):Wrapper(std::move(_env)),
                                              noop_max(noop_max),
                                              frame_skip(frame_skip),
                                              terminal_on_life_loss(terminal_on_life_loss),
                                              grayscale_obs(grayscale_obs),
                                              grayscale_new_axis(grayscale_new_axis),
                                              scale_obs(scale_obs),
                                              screen_size(screen_size) {
    assert(noop_max >= 0);
    assert(screen_size > 0);
    assert(frame_skip > 0);


    // buffer to store most recent for max pooling
    if (grayscale_obs)
        buffer = {torch::empty(env->ObservationSpace()->getShape(),
                               torch::TensorOptions(torch::kUInt8)),
                  torch::empty(env->ObservationSpace()->getShape(),
                               torch::TensorOptions(torch::kUInt8))};
    else
        buffer = {torch::empty(env->ObservationSpace()->getShape(), torch::TensorOptions(torch::kUInt8)),
                  torch::empty(env->ObservationSpace()->getShape(), torch::TensorOptions(torch::kUInt8))};

    shape = {grayscale_obs ? 1 : 3, screen_size, screen_size};
    if (grayscale_obs and not grayscale_new_axis)
        shape.pop_back();

    observation_space = not scale_obs ? std::make_unique<Box>(0, 255, shape, torch::kUInt8) :
                        std::make_unique<Box>(0, 1, shape, torch::kFloat32);
    action_space = std::move(env->action_space);
}

void AtariPreprocessing::fill_buffer(int idx)
{

    auto _buffer = dynamic_cast<BaseAtari*>(env.get())->get_screen(grayscale_obs);
    buffer[idx] = grayscale_obs? torch::from_blob(_buffer.data(),
                                                  buffer[idx].sizes(), torch::kUInt8) :
            buffer[idx] = torch::from_blob(_buffer.data(), buffer[idx].sizes(), torch::kUInt8);

}

torch::Tensor AtariPreprocessing::reset()
{
    env->reset();
    int noops = noop_max > 0 ? torch::randint(1, noop_max+1, {1}).item<int>() : 0;

    for(int i = 0 ; i < noops; i++)
    {
        auto res = env->step(torch::zeros({1}));
        if (res.done.item<bool>())
        {
            env->reset();
        }
    }

    lives = dynamic_cast<BaseAtari*>(env.get())->get_lives();
    fill_buffer(0);
    buffer[1] = torch::zeros_like(buffer[1], torch::kUInt8);
    return format_obs();
}


torch::Tensor AtariPreprocessing::format_obs()
{

    if(frame_skip > 1)
    {
        buffer[0] = torch::max(buffer[0], buffer[1]);

    }
    cv::Mat obs;
    auto type = grayscale_obs? CV_8UC1 : CV_8UC3;
    cv::Mat in = cv::Mat(buffer[0].size(0),
                         buffer[0].size(1),
                         type,buffer[0].data_ptr());
    cv::resize(in, obs, {screen_size, screen_size}, 0, 0, cv::INTER_AREA);

    torch::Tensor tensor_obs;
    if(scale_obs)
    {
        obs = obs/255;
        tensor_obs = torch::from_blob(obs.data, shape, torch::kFloat32);
    }else
        tensor_obs = torch::from_blob(obs.data, shape, torch::kUInt8);

    if(grayscale_obs && !grayscale_new_axis)
        tensor_obs = tensor_obs.squeeze(0);

    return tensor_obs;

}




StepResult AtariPreprocessing::step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action)
{
    float R = 0;

    bool done = false;
    float reward = 0;
    StepResult result;
    for(int t =0; t < frame_skip; t++)
    {
        result = env->step(action);
        R += result.rewards.item<float>();
        game_over = result.done.item<bool>();
        done = result.done.item<bool>();
        if (terminal_on_life_loss)
        {
            int new_lives =  dynamic_cast<BaseAtari*>(env.get())->get_lives();
            done = done || new_lives < lives;
            lives = new_lives;
        }

        if (done)
            break;

        if (t == frame_skip - 2)
        {
            fill_buffer(1);
        }
        else if (t == frame_skip - 1)
        {
            fill_buffer(0);
        }
    }
    return {format_obs(),
            torch::tensor(R),
            torch::tensor(done),
            result.info};
};

