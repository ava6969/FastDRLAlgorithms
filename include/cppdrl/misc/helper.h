//
// Created by dewe on 3/25/21.
//

#pragma once


#include <vector>
#include <cstdint>
#include <tuple>
#include "string"
#include "torch/torch.h"
#include "map"
#include <optional>
#include "boost/circular_buffer.hpp"
#include "variant"
#include "cppdrl/gym/spaces/space.h"
#define ALE_DIFFICULTIES {0, 1, 2, 3}
#define ALE_MODES {0, 1, 2, 3}
using std::string;
using std::vector;
using std::tuple;
using std::map;
using std::optional;
using std::variant;



enum class SampleBatchKey
{
    ACTIONS,
    LOGP,
    DONES,
    INFOS,
    NEXT_OBS,
    OBS,
    REWARDS,
    T,
    VF_PREDS,
    NEXT_VF_PREDS,
    ENTROPY,
    LOGITS,
    HIDDEN_STATE,
    ATTN_KEYS,
    RETURNS
};

enum class LossDictKey
{
    POLICY_LOSS,
    VALUE_LOSS,
    ENTROPY_LOSS,
    KL_LOSS
};

using ReplayQueue = map<SampleBatchKey, vector<vector<torch::Tensor>>>;
using SampleBatch = map<SampleBatchKey, torch::Tensor>;
using LossDict = map<LossDictKey, float>;
using ModelOutputDict = map<SampleBatch, torch::Tensor>;
using EnvConfig = map<string, variant<float, std::string, int>>;

struct ModelConfig
{
    optional<vector<torch::TensorOptions>> conv_args;
};

struct StepResult
{
    torch::Tensor observation, rewards, done;
    map<string, float> info;
};

struct TrainConfig
{
    size_t buffer_size=10000, rollout_length=500, max_target_update=100;
    unsigned long sgd_mini_batch{};
    size_t num_workers=1;
    Space* action_space;
    Space* observation_space;
    ModelConfig* modelConfig{};
    c10::Device device=torch::kCPU;
    EnvConfig env_config;
    int num_outputs;
    float start, end;
    double decay;
    size_t total_timesteps=100000;
    size_t update_freq=4;
    size_t report_every=10;
    size_t replay_start_size=1000;

    std::vector<SampleBatchKey> sample_keys{SampleBatchKey::OBS,
                                            SampleBatchKey::NEW_OBS,
                                            SampleBatchKey::DONES,
                                            SampleBatchKey::ACTIONS,
                                            SampleBatchKey::REWARDS};
};


enum class WorkerState
{
    IDLE, // means worker is waiting to be released
    RUNNING, // worker is running
    STARTED, // worker hasn't run yet but spawned
    STOPPED // worker is stopped
};


vector<int64_t> concat(torch::IntArrayRef const& op1, torch::IntArrayRef const& op2);
vector<int64_t>  concat(vector<int64_t> const& op1, vector<int64_t> const& op2);

