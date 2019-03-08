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

#define NUM_ITER 4
#define SIZE 10000000

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu(); // Use CPU for training

void fill(FAST::gam_vector<float> *in)
{
    in->resize(SIZE);
    for (auto item : *in)
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
                auto p_out = gam::public_ptr<FAST::gam_vector<float>>(out, [](FAST::gam_vector<float> *p_) { delete p_; });

                p_out.push(1);

                auto p_in = gam::pull_public<FAST::gam_vector<float>>(1);
                auto lp = p_in.unique_local();
                FAST_INFO("Iteration = " << i)
                // here end-of-scope triggers the destructor on the objects
            }
            break;
        }
        case 1:
        {
            for (int i = 0; i < NUM_ITER; i++)
            {
                auto p_in = gam::pull_public<FAST::gam_vector<float>>(0);
                auto lp = p_in.unique_local();
                FAST::gam_vector<float> *out = new FAST::gam_vector<float>();
                fill(out);
                auto p_out = gam::public_ptr<FAST::gam_vector<float>>(out, [](FAST::gam_vector<float> *p_) { delete p_; });

                p_out.push(0);
                FAST_INFO("Iteration = " << i)
                // here end-of-scope triggers the destructor on the objects
            }
            break;
        }
        }
    }
}
