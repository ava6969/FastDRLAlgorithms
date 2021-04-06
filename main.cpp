#include <iostream>
#include "torch/torch.h"
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include "cppdrl/cppdrl.h"

using std::cout;
using std::endl;
using namespace torch;
int main()
{

    CartPoleEnv test_env({});
    EnvConfig envConfig{};
    TrainConfig trainConfig;
    ModelConfig model_config;

    trainConfig.rollout_length = 100;
    trainConfig.max_target_update = 10;
    trainConfig.buffer_size = 50000;
    trainConfig.sgd_mini_batch = 64;
    trainConfig.num_workers = 1;
    trainConfig.action_space = test_env.Actionspace();
    trainConfig.observation_space = test_env.ObservationSpace();
    trainConfig.modelConfig =&model_config;
    trainConfig.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    trainConfig.env_config = envConfig;
    trainConfig.num_outputs = test_env.Actionspace()->getShape()[0];
    trainConfig.decay = 20000;
    trainConfig.start = 1;
    trainConfig.end = 0.3;
    trainConfig.total_timesteps = 10000000000;
    trainConfig.update_freq = 1;
    trainConfig.report_every = 1000;
    trainConfig.replay_start_size = 5 * 64;

    torch::manual_seed(12);
    torch::cuda::manual_seed_all(12);

    DQNTrainer<MLP,
    torch::optim::RMSprop,
    torch::optim::RMSpropOptions, CartPoleEnv, EpsilonStrategy>
    trainer{trainConfig,torch::optim::RMSpropOptions().lr(0.0005)};
    trainer.train();


}