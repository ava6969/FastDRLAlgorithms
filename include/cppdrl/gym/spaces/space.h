//
// Created by dewe on 3/25/21.
//

#pragma once

#include <utility>
#include <variant>
#include <random>
#include "vector"
#include <torch/torch.h>

using std::variant;
using std::vector;

class Space
{
protected:
    vector<int64_t> shape;
    torch::ScalarType dtype;

public:
    Space(vector<int64_t> const& shape, torch::ScalarType dtype):shape(shape), dtype(dtype) {}

    virtual torch::Tensor sample() = 0;

    virtual bool operator == (Space const& other) const
    {
        return this->shape[0] == other.shape[0];
    }

    virtual bool operator != (Space const& other) const
    {
        return this->shape[0] != other.shape[0];
    }

    [[nodiscard]] virtual vector<int64_t>  getShape() const { return this->shape; }

    torch::ScalarType getType() const { return this->dtype; }

    virtual ~Space() = default;

    virtual std::string toString() = 0;

};


