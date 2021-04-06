//
// Created by dewe on 3/27/21.
//

#pragma once

#include <cppdrl/gym/spaces/discrete.h>
#include <cppdrl/models/nn_base.h>

class MLP : public NNBase{

private:
    nn::Sequential model;

public:
    MLP(TrainConfig const& config):
                  model(nn::Linear(nn::LinearOptions(config.observation_space->getShape()[0], 512)),            // 20 * 20
                        nn::ReLU(),
                        nn::Linear(nn::LinearOptions(512, 256)),// 9 * 9
                         nn::ReLU(),
                         nn::Linear(nn::LinearOptions(256, config.num_outputs)))
    {
        register_module("linear_model", model);
//        init_weights(model->named_parameters(), sqrt(2.), 0);
    }

    ModelOutputDict forward(SampleBatch const& batch, Tensor const& state) override;

};
