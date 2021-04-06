//
// Created by dewe on 3/26/21.
//

#ifndef DRLALGORITHMS_NN_BASE_H
#define DRLALGORITHMS_NN_BASE_H

#include <cppdrl/gym/spaces/space.h>
#include "torch/torch.h"
#include "cppdrl/misc/helper.h"
#include "map"
#include "variant"

using std::map;

using namespace torch;

class NNBase : public nn::Module {

protected:

public:
    NNBase()=default;

    virtual ModelOutputDict forward(SampleBatch const& batch, Tensor const& state) = 0;

    void init_weights(const torch::OrderedDict<std::string, torch::Tensor>& parameters,
                      double weight_gain,
                      double bias_gain)
    {
        for (const auto &parameter : parameters)
        {
            if (parameter.value().size(0) != 0)
            {
                if (parameter.key().find("bias") != std::string::npos)
                {
                    nn::init::constant_(parameter.value(), bias_gain);
                }
                else if (parameter.key().find("weight") != std::string::npos)
                {
                    orthogonal_(parameter.value(), weight_gain);
                }
            }
        }
    }

    static torch::Tensor orthogonal_(Tensor tensor, double gain)
    {
        NoGradGuard guard;

        BOOST_VERIFY_MSG(
                tensor.ndimension() >= 2,
                "Only tensors with 2 or more dimensions are supported");

        const auto rows = tensor.size(0);
        const auto columns = tensor.numel() / rows;
        auto flattened = torch::randn({rows, columns});

        if (rows < columns)
        {
            flattened.t_();
        }

        // Compute the qr factorization
        Tensor q, r;
        std::tie(q, r) = torch::qr(flattened);
        // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        auto d = torch::diag(r, 0);
        auto ph = d.sign();
        q *= ph;

        if (rows < columns)
        {
            q.t_();
        }

        tensor.view_as(q).copy_(q);
        tensor.mul_(gain);

        return tensor;
    }
};

#endif //DRLALGORITHMS_NN_BASE_H
