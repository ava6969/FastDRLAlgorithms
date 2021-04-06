//
// Created by dewe on 3/27/21.
//

#include "cppdrl/misc/helper.h"

vector<int64_t>  concat(vector<int64_t> const& op1, vector<int64_t> const& op2)
{
    vector<long> shape;
    shape.insert(std::end(shape), std::begin(op1), std::end(op1));
    shape.insert(std::end(shape), std::begin(op2), std::end(op2));

    return shape;
}



vector<int64_t> concat(torch::IntArrayRef const& op1, torch::IntArrayRef const& op2)
{
    vector<long> shape;
    shape.insert(std::end(shape), std::begin(op1), std::end(op1));
    shape.insert(std::end(shape), std::begin(op2), std::end(op2));

    return shape;
}