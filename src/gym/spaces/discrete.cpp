//
// Created by dewe on 3/27/21.
//

#include <cppdrl/gym/spaces/discrete.h>


ostream& operator<<(ostream& os, const Discrete& dt)
{
    os << "Discrete(" << dt.n << ")";
    return os;
}
