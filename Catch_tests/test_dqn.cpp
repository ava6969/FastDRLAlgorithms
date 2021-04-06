#define CATCH_CONFIG_MAIN // provides main(); this line is required in only one .cpp file
#include "catch.hpp"
#include "torch/torch.h"

#define private public
#define protected public
#include "cppdrl/cppdrl.h"

using DQNType = DQN<MLP, torch::optim::RMSprop, torch::optim::RMSpropOptions, EpsilonStrategy>;

void is_equal(DQNType* t1)
{
    for(int i = 0; i < t1->target_model->parameters().size(); i++)
    {
        auto p1 = t1->target_model->parameters()[i];
        auto p2 = t1->model->parameters()[i];

        auto res = p1.data().ne(p2.data()).sum().item<float>();
        REQUIRE(res <= 0);
    }
}

TEST_CASE( "Evaluate DQN Algorithm Complete" )
{
    CartPoleGym test_env({});
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
    trainConfig.device = torch::kCUDA;
    trainConfig.env_config = envConfig;
    trainConfig.num_outputs = test_env.Actionspace()->getShape()[0];
    trainConfig.decay = 2 * 1e-4;
    trainConfig.start = 1;
    trainConfig.end = 0.3;
    trainConfig.total_timesteps = 10000000000;
    trainConfig.update_freq = 5;
    trainConfig.report_every = 1000;
    trainConfig.replay_start_size = 5 * 64;
    DQNTrainer<MLP, torch::optim::RMSprop,torch::optim::RMSpropOptions,CartPoleGym, EpsilonStrategy>
                _trainer{trainConfig,
                        torch::optim::RMSpropOptions()};

    auto dqn_policy = dynamic_cast<DQNType*>(_trainer.trainer.get());
    int index = 0;
    EnvConfig env_config = trainConfig.env_config;
    env_config["worker_id"] = index +  1;
    _trainer.worker_envs[index] = std::make_unique<CartPoleGym>(env_config);
    _trainer.worker_policies[index] = _trainer.create_policy();
    auto& worker_env =  _trainer.worker_envs[index];
    auto* es = dynamic_cast<EpsilonStrategy*>(dynamic_cast<DQNType*>(_trainer.trainer.get())->strategy.get());
    StepResult result;

    SECTION("Test Strategy Init")
    {

        CHECK(es->decay == trainConfig.decay);
        CHECK(es->start == trainConfig.start);
        CHECK(es->end == trainConfig.end);
        CHECK(es->epsilon == trainConfig.start);

    }


    SECTION("Test Models initialization")
    {
        REQUIRE(dqn_policy->target_model);
        REQUIRE(dqn_policy->model);
        REQUIRE(dqn_policy->optimizer);

        is_equal(dqn_policy);

    }

    SECTION("Test Environment step")
    {

        auto observation = worker_env->reset();
        cout << "first observation: " << observation << endl;
        std::tie(observation, std::ignore, result) = _trainer.step(observation, torch::Tensor{}, index, 0);
        cout << "second observation: " << observation << endl;
        cout << "action_taken: " << _trainer.replay_queue[SampleBatchKey::ACTIONS][index][0] << endl;
        cout << "reward gotten: " << _trainer.replay_queue[SampleBatchKey::REWARDS][index][0] << endl;


        std::tie(observation, std::ignore, result) = _trainer.step(observation, torch::Tensor{}, index, 0);
        cout << "third observation: " << observation << endl;
        cout << "action_taken: " << _trainer.replay_queue[SampleBatchKey::ACTIONS][index][0] << endl;
        cout << "reward gotten: " << _trainer.replay_queue[SampleBatchKey::REWARDS][index][0] << endl;

    }


}