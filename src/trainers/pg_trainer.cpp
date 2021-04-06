//
// Created by dewe on 3/30/21.
//

#include "cppdrl/trainers/pg_trainer.h"
#include "cppdrl/models/nn_base.h"
#include "cppdrl/policies/policy.h"
using std::begin;
using std::endl;
using std::cout;
using std::end;

SampleBatch PGTrainer::getTrainBatch()
{
    auto obs_shape = gymEnv->ObservationSpace()->getShape();
    auto action_shape = gymEnv->Actionspace()->getShape();

    auto cat_obs_shape = concat({long(n_workers * max_fragment_size * rollout_fragment_length)}, obs_shape);
    auto cat_action_shape = concat({long(n_workers * max_fragment_size * rollout_fragment_length)}, action_shape);

    SampleBatch sampleBatch;

    auto obs = torch::cat(buffer[SampleBatchKey::OBS], 0);
    sampleBatch[SampleBatchKey::OBS] = obs.view(cat_obs_shape);

    auto n_obs = torch::cat(buffer[SampleBatchKey::NEW_OBS], 0);
    sampleBatch[SampleBatchKey::NEW_OBS]  = n_obs.view(cat_obs_shape);

    auto action = torch::cat(buffer[SampleBatchKey::ACTIONS], 0);
    sampleBatch[SampleBatchKey::ACTIONS]  = action.view(cat_action_shape);

    return sampleBatch;
}