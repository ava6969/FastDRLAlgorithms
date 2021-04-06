//
// Created by dewe on 3/25/21.
//

#pragma once

#include "space.h"
#include <iostream>

using std::ostream;

class MultiDiscrete: public Space
{
    /**A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    Example::
    >>> Discrete(2)
    **/

public:
    const vector<uint64_t>  n_vec;

    explicit MultiDiscrete(vector<uint64_t> const& n_vec):Space({static_cast<long>(n_vec.size())},
                                                                torch::kInt32),n_vec(n_vec) {

    }

    friend ostream& operator << (ostream& os, const MultiDiscrete& dt);

    virtual bool operator == (MultiDiscrete const& other) const
    {
        return n_vec == other.n_vec;
    }

    virtual bool operator != (MultiDiscrete const& other) const
    {
        return n_vec != other.n_vec;
    }
    torch::Tensor sample() override {

        vector<torch::Tensor> tensors(n_vec.size());
        std::transform(begin(n_vec), end(n_vec), begin(tensors),[](uint64_t dim){
            return torch::randint(dim, {1});
        });

        return torch::cat(tensors);
    }

    std::string toString() override
    {
        std::stringstream ss;
        ss << "MultiDiscrete(" << n_vec << ")";
        return ss.str();
    }

};

ostream& operator << (ostream& os, const MultiDiscrete& dt);