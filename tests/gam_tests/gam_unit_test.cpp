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
		vector<int> ref = {1,2,3};
		switch (gam::rank()) {
		case 0:
		{
			// check local consistency on public pointer
			std::shared_ptr<FAST::gam_vector<int>> lp = nullptr;
			{
				auto p = gam::make_public<FAST::gam_vector<int>>(ref);
				lp = p.local();
				// here end-of-scope triggers the destructor on the original object
			}
			assert(*lp == ref);

			auto p = gam::make_public<FAST::gam_vector<int>>(ref);
			p.push(1);

			auto q = gam::make_private<FAST::gam_vector<int>>(ref);
			q.push(1);
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
			assert(*lp == ref);

			auto q = gam::pull_private<FAST::gam_vector<int>>(0);
			auto lq = q.local();
			assert(*lq == ref);
			break;
		}
		}
	}
}
