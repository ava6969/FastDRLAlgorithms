//
// Created by dewe on 3/29/21.
//

#ifndef CPPDRL_ACTIONDISTRIBUTION_H
#define CPPDRL_ACTIONDISTRIBUTION_H

#include "torch/torch.h"
class ActionDistribution {

protected:
    torch::Tensor logits{};

public:
    ActionDistribution(torch::Tensor const& _logits):logits(_logits){}

    virtual torch::Tensor sample() = 0;

    virtual torch::Tensor deterministic_sample() = 0;
};


#endif //CPPDRL_ACTIONDISTRIBUTION_H
