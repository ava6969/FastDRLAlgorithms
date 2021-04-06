//
// Created by dewe on 3/29/21.
//

#ifndef CPPDRL_STRATEGY_H
#define CPPDRL_STRATEGY_H

#include "cppdrl/misc/helper.h"
#include <torch/torch.h>

struct Strategy {
protected:
    float start, end;
    double decay;
    class Space* action_space;
public:
    Strategy(TrainConfig const& trainConfig): start(trainConfig.start),
    end(trainConfig.end),
    decay(trainConfig.decay),
    action_space(trainConfig.action_space)
    {

    }

    virtual ModelOutputDict act(class NNBase* model, SampleBatch const& sample_batch, const torch::Tensor& state) = 0;

    virtual ~Strategy()=default;
};

struct EpsilonStrategy: public Strategy {
protected:
    double epsilon{start}, t;
    vector<double> epsilons;
public:
    EpsilonStrategy(TrainConfig const& trainConfig): Strategy(trainConfig)
    {
        auto e = (0.01/ torch::logspace(-2, 0, decay)) - 0.01;
        e = e * (start - end) + end;
        e = e.toType(torch::kDouble);
        epsilons.resize(e.size(0));
        epsilons.reserve(e.size(0));
        memcpy(epsilons.data(), e.data_ptr<double>(), e.size(0) * sizeof(double));

    }
    ModelOutputDict act(class NNBase* model, const SampleBatch& sample_batch, const torch::Tensor& state) override;
};


#endif //CPPDRL_STRATEGY_H
