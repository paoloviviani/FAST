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

#ifndef FAST_TESTLOG
#define FAST_TESTLOG(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_error() << \
		"TEST: " << Catch::getResultCapture().getCurrentTestName() << " -- " << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}

#define FAST_TESTNAME FAST_TESTLOG("")
#endif

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu();  // Use CPU for training

TEST_CASE( "Gam basic", "gam" ) {
	auto p = gam::make_private<int>(42);
	assert(p != nullptr);
	auto local_p = p.local();
	REQUIRE(*local_p == 42);
}

TEST_CASE( "Gam private vector", "gam" ) {
	auto data = gam::make_private<vector<float>>(3);
	vector<float> buf = {11.,12.,13.};
	auto local_data = data.local();
	*local_data = buf;
	data = gam::private_ptr< vector<float>>(std::move(local_data));
}

TEST_CASE( "Tensor with Gam data", "gam,tensor" ) {
	vector<unsigned int> shape = {2,3};
	vector<float> data = {11.,12.,13.,21.,22.,23.};

	FAST::Tensor<NDArray> tensor(data.data(),shape);
	REQUIRE(tensor.getShape() == shape);
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.at(i,j) == data.at(i*shape[1]+j));
		}
}

TEST_CASE( "SPMD Moving tensor", "gam,tensor" ) {
	if (FAST::cardinality() > 1) {
		switch (FAST::rank())
	    {
	    case 0:
			vector<unsigned int> shape = {2,3};
			vector<float> data = {11.,12.,13.,21.,22.,23.};
			FAST::Tensor<NDArray> tensor(data.data(),shape);
	        auto p = tensor.getPrivatePtr();
			/* push to 1 */
	        p.push(1);

	        /* wait for the response */
	        p = gam::pull_private<vector<float>>(1);
	        assert(*p.local() == data);
	        break;
	    case 1:
	        /* pull private pointer from 0 */
	        auto p = gam::pull_private<vector<float>>(); //from-any just for testing

	        /* push back to 0 */
	        p.push(0);
	        break;
	    }
	}
}
