//
// Created by dewe on 3/27/21.
//

#include <cppdrl/gym/spaces/box.h>

ostream& operator<<(ostream &os, const Box &box) {
    os << "Box(" << box.low << "," <<  box.high << "," << box.shape << ", " << box.dtype << ")";
    return os;
}

