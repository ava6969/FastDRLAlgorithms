//
// Created by dewe on 3/27/21.
//

#ifndef DRLALGORITHMS_CNN_NATURE_H
#define DRLALGORITHMS_CNN_NATURE_H

#include <cppdrl/gym/spaces/discrete.h>
#include <cppdrl/models/nn_base.h>

class CNNNature : public NNBase{

private:
    nn::Sequential conv_model;

public:
    CNNNature(TrainConfig const& config):conv_model(nn::Conv2d(nn::Conv2dOptions(config.observation_space->getShape()[0], 32, 8).stride(4)),            // 20 * 20
                                     nn::ReLU(),
                                     nn::Conv2d(nn::Conv2dOptions(32, 64, 4).stride(2)),// 9 * 9
                                     nn::ReLU(),
                                     nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1)),// 10 * 10
                                     nn::Flatten(),
                                     nn::Linear(7 * 7 * 64, 512),
                                     nn::ReLU(),
                                     nn::Linear(512, config.num_outputs))
    {
        register_module("conv_model", conv_model);
        init_weights(conv_model->named_parameters(), sqrt(2.), 0);
    }

    ModelOutputDict forward(SampleBatch const& batch, Tensor const& state) override;

};


#endif //DRLALGORITHMS_CNN_NATURE_H
