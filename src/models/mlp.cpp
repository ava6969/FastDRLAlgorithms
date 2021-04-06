//
// Created by dewe on 3/27/21.
//

#include "cppdrl/models/mlp.h"

ModelOutputDict MLP::forward(const SampleBatch &batch, const Tensor &state)
{
    return {{ModelOutputKey::LOGITS , model->forward(batch.at(SampleBatchKey::OBS))},
            {ModelOutputKey::HIDDEN_STATE, state}};

}