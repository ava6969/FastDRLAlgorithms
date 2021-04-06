//
// Created by dewe on 3/29/21.
//
#include "cppdrl/models/cnn_nature.h"

ModelOutputDict CNNNature::forward(const SampleBatch &batch, const Tensor &state)
{
    torch::Tensor obs = batch.at(SampleBatchKey::OBS);
    if(batch.at(SampleBatchKey::OBS).sizes().size() == 3)
        obs = batch.at(SampleBatchKey::OBS).unsqueeze(0);

    return {{ModelOutputKey::LOGITS , conv_model->forward(obs)},
            {ModelOutputKey::HIDDEN_STATE, state}};

}