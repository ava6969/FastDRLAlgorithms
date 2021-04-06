//
// Created by dewe on 3/27/21.
//

#pragma once

#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <array>
#include <cppdrl/gym/base_wrappers.h>

class BaseAtari;
using std::array;

class AtariPreprocessing: public Wrapper {
    /**Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
            * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
                             * Scale observation: optional
            Args:
            env (Env): environment
    noop_max (int): max number of no-ops
    frame_skip (int): the frequency at which the agent experiences the game.
    screen_size (int): resize Atari frame
    terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
    life is lost.
    grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
    grayscale_newaxis (bool): if True and grayscale_obs=True, then a channel axis is added to
    grayscale observations to make them 3-dimensional.
    scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
    optimization benefits of FrameStack Wrapper.
    */

private:

    int noop_max, frame_skip, screen_size, terminal_on_life_loss, grayscale_obs, grayscale_new_axis, scale_obs, lives{};
    array<torch::Tensor, 2> buffer;
    bool game_over = false;
    vector<long> shape;

public:
    AtariPreprocessing(unique_ptr<BaseAtari> env,
                       const int noop_max=30,
                       const int frame_skip=4,
                       const int screen_size=84,
                       const bool terminal_on_life_loss=false,
                       const bool grayscale_obs=true,
                       const bool grayscale_new_axis=true,
                       const bool scale_obs=true);

    void fill_buffer(int idx);

    torch::Tensor reset() override;

    torch::Tensor format_obs();

    std::optional<cv::Mat> render(bool human) override { return env->render(human);}

    StepResult step(std::variant<torch::Tensor, std::map<std::string, torch::Tensor>> const& action) override;

};

