//
// Created by dewe on 3/28/21.
//

#include "cppdrl/gym/spaces/multi_discrete.h"

ostream& operator << (ostream& os, const MultiDiscrete& dt)
{
    os << "MultiDiscrete(" << dt.n_vec << ")";
    return os;
}