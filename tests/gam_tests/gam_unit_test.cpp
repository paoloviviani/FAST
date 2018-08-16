/*
 * example_unit_test.cpp
 *
 *  Created on: May 30, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include "fast.hpp"
#include "mxnet-cpp/MxNetCpp.h"
#include "gam.hpp"
#include <string>

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu();  // Use CPU for training

TEST_CASE( "Gam basic", "gam" ) {
	FAST_LOG_INIT
	auto p = gam::make_private<int>(42);
	assert(p != nullptr);
	auto local_p = p.local();
	REQUIRE(*local_p == 42);
}

TEST_CASE( "SPMD tensor send", "gam,tensor" ) {
	if (gam::cardinality() > 1) {
		vector<unsigned int> shape = {2,3};
		vector<float> data = {11.,12.,13.,21.,22.,23.};
		switch (gam::rank()) {
		case 0:
		{
			FAST::Tensor<float> tensor(data.data(),shape);
			tensor.push(1);
			break;
		}
		case 1:
		{
			auto recv = FAST::pull_tensor<float>();
			recv->reShape(shape);
			for (uint i = 0; i < shape[0]; i++)
				for (uint j = 0; j < shape[1]; j++){
					REQUIRE(recv->at(i,j) == data.at(i*shape[1]+j));
				}
			break;
		}
		}
	}
}

TEST_CASE( "SPMD tensor ping-pong", "gam,tensor" ) {
	if (gam::cardinality() > 1) {
		vector<unsigned int> shape = {2,3};
		vector<float> data = {11.,12.,13.,21.,22.,23.};
		switch (gam::rank()) {
		case 0:
		{
			FAST::Tensor<float> tensor(data.data(),shape);
			tensor.push(1);
			auto recv = FAST::pull_tensor<float>();
			recv->reShape(shape);
			for (uint i = 0; i < shape[0]; i++)
				for (uint j = 0; j < shape[1]; j++){
					REQUIRE(recv->at(i,j) == data.at(i*shape[1]+j));
				}
			break;
		}
		case 1:
		{
			auto recv = FAST::pull_tensor<float>();
			recv->reShape(shape);
			recv->push(0);
			break;
		}
		}
	}
}

TEST_CASE( "SPMD public vector ping-pong", "gam,vector,public" ) {
	if (gam::cardinality() > 1) {
		switch (gam::rank()) {
		case 0:
		{
		    auto p = gam::make_private<FAST::gam_vector<int>>();

		    /* populate */
		    auto lp = p.local();
		    lp->push_back(1);
		    lp->push_back(2);
		    lp->push_back(3);
		    p.push(1);
			break;
		}
		case 1:
		{
			 auto p = gam::pull_public<FAST::gam_vector<int>>(); //from-any just for testing

			    /* test and add */
			    auto lp = p.local();
			    assert(lp->size() == 3);
			    assert(lp->at(1) == 2);
			break;
		}
		}
	}
}
