//
// Created by dewe on 3/27/21.
//

#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <iostream>

using std::begin;
using std::end;

#define private public
#include <cppdrl/cppdrl.h>

BOOST_AUTO_TEST_SUITE(Shapes_suite)

    BOOST_AUTO_TEST_CASE(discrete)
    {
        Discrete ds{4};
        Discrete ds1{4};
        Discrete ds2{20};
        std::cout << ds << std::endl;
        std::cout << "discrete: sample:" << ds.sample() << std::endl;
        BOOST_TEST(ds.sample()[0].item<int>() >= 0);
        BOOST_TEST(ds.sample()[0].item<int>() <= 4);
        BOOST_TEST(ds.n == 4);

        std::cout << "discrete: shape:" << ds1.getShape()[0] << std::endl;
        BOOST_TEST(ds != ds2);
        BOOST_TEST(ds == ds1);
        BOOST_TEST(ds.getType() == c10::kInt);
    }

    BOOST_AUTO_TEST_CASE(box)
    {
        Box box1{0, 255, {3, 3, 3}, torch::kInt};
        Box box2{0, 1, {3, 3}, torch::kFloat32};
        Box box3{0, 255, {3}, torch::kUInt8};

        auto box_test = [&](Box& box, torch::ScalarType type, float range) {

            std::cout << box << std::endl;
            auto sample = box.sample();
            std::cout <<"Box Sample:" << sample << std::endl;
//            torch::less_equal(sample, range);
            //      todo: fix sample() for float type
            std::cout <<"Box Sample Sizes:" << sample.sizes() << std::endl;
            std::cout <<"Box Sample Shape:" << box.getShape() << std::endl;
            BOOST_TEST(sample.sizes()[0] == box.getShape()[0]);
            BOOST_TEST(box.getType() == type);

        };

        box_test(box1, torch::kInt, 255);
        box_test(box2, torch::kFloat32, 1);
        box_test(box3, torch::kUInt8, 255);
    }

    BOOST_AUTO_TEST_CASE(multi_discrete)
    {
        MultiDiscrete ds{{4, 34, 6}};

        std::cout << ds << std::endl;
        std::cout << "multi discrete: sample:" << ds.sample() << std::endl;
        auto sampled = ds.sample();
        BOOST_CHECK(sampled[0].item<int>() >= 0 && sampled[0].item<int>() < 4 );
        BOOST_CHECK(sampled[1].item<int>() >= 0 && sampled[1].item<int>() < 34 );
        BOOST_CHECK(sampled[2].item<int>() >= 0 && sampled[2].item<int>() < 6 );

        BOOST_TEST(sampled.sizes()[0] == 3);
        BOOST_TEST(ds.getType() == c10::kInt);
    }
    // End of test suite
BOOST_AUTO_TEST_SUITE_END()




