// Created by dewe on 3/25/21.
//

#pragma once

#include "space.h"
#include "iostream"
#include "sstream"

using std::ostream;

class Box : public Space
{
private:
    double low, high;

public:
    Box(double low, double high, vector<int64_t> const&  shape, torch::ScalarType type):Space(shape, type),
    low(low), high(high)
    {}

    torch::Tensor sample() override
    {
        if (dtype == torch::kFloat)
        {
            torch::Tensor t = torch::rand(shape);
            t = t.mul(high - low) + low;
            return t.toType(dtype);
        }

        else
            return torch::randint(this->low, this->high,shape).toType(dtype);

    }

    std::string toString() override
    {
        std::stringstream ss;
        ss << "Box(" << low << "," <<  high << "," << shape << dtype << ")";
        return ss.str();
    }

    friend ostream& operator<<(ostream& os, const Box& box);
};
ostream& operator<<(ostream& os, const Box& box);