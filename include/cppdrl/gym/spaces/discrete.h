//
// Created by dewe on 3/25/21.
//

#pragma once

#include "space.h"
#include <iostream>

using std::ostream;

class Discrete: public Space
{
    /**A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    Example::
    >>> Discrete(2)
    **/

public:
    const int64_t n;

    Discrete(int64_t _n):Space({_n}, torch::kInt32),n(_n) { }

    friend ostream& operator << (ostream& os, const Discrete& dt);

    virtual bool operator == (Discrete const& other) const
    {
        return n == other.n;
    }

    virtual bool operator != (Discrete const& other) const
    {
        return n != other.n;
    }
    torch::Tensor sample() { return torch::randint(n, {1}); }

    std::string toString() override
    {
        std::stringstream ss;
        ss << "Discrete(" << n << ")";
        return ss.str();
    }

};

ostream& operator << (ostream& os, const Discrete& dt);