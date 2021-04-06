//
// Created by dewe on 3/29/21.
//
#include "cppdrl/misc/strategy.h"
#include "cppdrl/gym/spaces/space.h"
#include "cppdrl/models/nn_base.h"

ModelOutputDict EpsilonStrategy::act(class NNBase *model, const SampleBatch &sample_batch, const torch::Tensor &state) {
    torch::Tensor action;
    ModelOutputDict modelOutputDict;
    if(torch::rand({1}).item<float>() > epsilon ) // anneal epsilon
    {
        {
            torch::NoGradGuard guard;
            modelOutputDict = model->forward(sample_batch, state);
            action = std::get<1>(torch::max(modelOutputDict[ModelOutputKey::LOGITS].detach(), -1)).unsqueeze(0);
        }

    } else
    {
        action = action_space->sample();
    }

    epsilon = t > decay ? end : epsilons[t];
    t++;
    modelOutputDict[ModelOutputKey::ACTION] = action;
    return modelOutputDict;
}
