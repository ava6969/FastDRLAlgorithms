//
// Created by dewe on 3/28/21.
//
#include <boost/test/unit_test.hpp>

#define private public
#include <cppdrl/cppdrl.h>

BOOST_AUTO_TEST_SUITE(Helper)

BOOST_AUTO_TEST_CASE(concat_function)
        {
                Box box1{0, 255, {3, 84, 84}, torch::kInt};
        Box box2{0, 1, {2, 84, 84}, torch::kFloat32};
        Box box3{0, 1, {84, 84}, torch::kFloat32};

        auto res = concat(box1.getShape(), box2.getShape());
        auto true_res = {3, 84, 84, 2, 84, 84};
        BOOST_CHECK_EQUAL_COLLECTIONS(res.begin(), res.end(),
        true_res.begin(), true_res.end());


        auto res1 = concat(box1.getShape(), box3.getShape());
        auto true_res1 = {3, 84, 84, 84, 84};
        BOOST_CHECK_EQUAL_COLLECTIONS(res1.begin(), res1.end(),
        true_res1.begin(), true_res1.end());
        }

BOOST_AUTO_TEST_SUITE_END()