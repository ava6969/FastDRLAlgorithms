//
// Created by dewe on 3/28/21.
//
#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <iostream>

using std::begin;
using std::end;

#define private public
#include <cppdrl/cppdrl.h>
#include <cppdrl/gym/ml_pack_wrapper.h>


BOOST_AUTO_TEST_SUITE(envs)

    BOOST_AUTO_TEST_CASE(cartpole_test_loop)
            {
                GymEnv* env = new CartPoleGym();

                int i = 4;
                while(i-- > 0)
                {
                    auto obs = env->reset();
                   while(true)
                   {
                       auto action = env->Actionspace()->sample();
                       auto result = env->step(action);
                       if(result.done.item<bool>())
                       {
                           break;
                       }
                   }
                }

                delete env;
            }

BOOST_AUTO_TEST_SUITE_END()