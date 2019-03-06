/*
 * example_unit_test.cpp
 *
 *  Created on: May 30, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include "fast.hpp"
#include <string>

#define NUM_ITER 100
#define SIZE 10000000

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu(); // Use CPU for training

void fill(FAST::gam_vector<float> *in)
{
    in.resize(SIZE);
    for (auto item : in)
        item = 1.;
}

TEST_CASE("SPMD public vector multiple ping-pong", "gam,vector,public")
{
    FAST_INFO("TEST name: " << Catch::getResultCapture().getCurrentTestName());

    if (gam::cardinality() > 1)
    {
        vector<int> ref = {1, 2, 3};
        switch (gam::rank())
        {
        case 0:
        {
            for (int i = 0; i < NUM_ITER; i++)
            {
                FAST::gam_vector<float> *out = new FAST::gam_vector<float>();
                fill(out);
                auto p = gam::public_ptr<FAST::gam_vector<float>>(out, [](float *p_) { delete p_; });

                p.push(1);

                std::shared_ptr<FAST::gam_vector<float>> lp = nullptr;
                {
                    auto p = gam::pull_public<FAST::gam_vector<float>>(1);
                    lp = p.local();
                    // here end-of-scope triggers the destructor on the original object
                }
            }
            break;
        }
        case 1:
        {
            std::shared_ptr<FAST::gam_vector<int>> lp = nullptr;
            {
                auto p = gam::pull_public<FAST::gam_vector<int>>(0);
                lp = p.local();
                // here end-of-scope triggers the destructor on the original object
            }
            REQUIRE(*lp == ref);

            auto q = gam::pull_private<FAST::gam_vector<int>>(0);
            auto lq = q.local();
            REQUIRE(*lq == ref);
            break;
        }
        }
    }
}
