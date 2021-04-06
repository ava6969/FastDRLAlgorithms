//
// Created by dewe on 3/29/21.
//

#ifndef CPPDRL_TRAINER_H
#define CPPDRL_TRAINER_H


#include <cstddef>
#include "thread"
#include "vector"
#include "torch/torch.h"
#include <string>
#include <map>
#include <cppdrl/policies/policy.h>
#include "variant"
#include "cppdrl/misc/helper.h"
#include "cppdrl/models/nn_base.h"

const int LAST_100_EPISODIC_REW = 100;
using std::map;
using std::string;
using std::vector;
using std::variant;
using std::unique_ptr;
class NNBase;
class GymEnv;



//todo: make rollout sampler abstract - truncated or complete episode

template<typename ModelType,
        typename OptimizerType,
        typename OptimizerOption,
        typename EnvType>
class Trainer {

protected:

    TrainConfig train_config;

    unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>> trainer;

    vector< unique_ptr<GymEnv> > worker_envs{train_config.num_workers};
    vector<std::thread> worker_threads{train_config.num_workers};
    vector<std::condition_variable> cv_s{train_config.num_workers};
    vector<std::mutex> mutexes{train_config.num_workers};
    vector<unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>>> worker_policies{train_config.num_workers};
    vector<WorkerState> worker_states{train_config.num_workers};

    Tensor last_100_reward;
    torch::Tensor time_step_tensor;

public:
    explicit Trainer(TrainConfig const& train_config):train_config(train_config),
    time_step_tensor(torch::zeros({long(train_config.num_workers), 1}, torch::TensorOptions(torch::kI32))),
    last_100_reward(torch::zeros({long(train_config.num_workers), LAST_100_EPISODIC_REW}, torch::TensorOptions(torch::kFloat32)))
    {
    }

    void start()
    {
        for(int i = 0; i < train_config.num_workers; i++)
        {
            worker_states[i] = WorkerState::STARTED;
            worker_threads[i] = std::thread(&Trainer::run, this, i);
            EnvConfig env_config = train_config.env_config;
            env_config["worker_id"] = i + 1;
            worker_envs[i] = std::make_unique<EnvType>(env_config);
            worker_policies[i] = create_policy();
        }
    }

    void stop()
    {
        for(int i = 0 ; i < train_config.num_workers; i++)
        {
            worker_states[i] = WorkerState::STOPPED;
            worker_threads[i].join();
        }
    }

    void awake()
    {
        for(int i = 0; i < train_config.num_workers; i++)
        {
            worker_states[i] = WorkerState::RUNNING;
            cv_s[i].notify_one();
        }
    }

    virtual unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>> create_policy() = 0;
    virtual void run(int index) = 0;
    virtual SampleBatch getTrainBatch() = 0;
    virtual void train() = 0;

    virtual void transport_weights()
    {
        for(unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>>& policy: worker_policies)
        {
            policy->update_weight(trainer->get_model());
        }
    }


};


#endif //CPPDRL_TRAINER_H
