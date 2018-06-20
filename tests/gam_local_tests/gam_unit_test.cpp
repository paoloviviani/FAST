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
			FAST::Tensor<NDArray> tensor(data.data(),shape);
			tensor.push(1);
			break;
		}
		case 1:
		{
			auto p = gam::pull_private<gam_vector<float>>();
			auto p_local = p.local();
			FAST::Tensor<NDArray> tensor(p_local->data(),shape);
			for (uint i = 0; i < shape[0]; i++)
				for (uint j = 0; j < shape[1]; j++){
					REQUIRE(tensor.at(i,j) == data.at(i*shape[1]+j));
				}
			break;
		}
		}
	}
}

//TEST_CASE( "SPMD tensor ping-pong", "gam,tensor" ) {
//	if (gam::cardinality() > 1) {
//		switch (gam::rank()) {
//		case 0:
//		{
//			/* create a private pointer */
//			auto p = gam::make_private<gam_vector<int>>();
//
//			/* populate */
//			auto lp = p.local();
//			lp->push_back(42);
//
//			/* push to 1 */
//			p = gam::private_ptr<gam_vector<int>>(std::move(lp));
//			p.push(1);
//
//			/* wait for the response */
//			p = gam::pull_private<gam_vector<int>>(1);
//
//			/* test */
//			lp = p.local();
//			REQUIRE(lp->size() == 10);
//			REQUIRE(lp->at(1) == 43);
//			break;
//		}
//		case 1:
//		{
//			/* pull private pointer from 0 */
//			auto p = gam::pull_private<gam_vector<int>>(); //from-any just for testing
//
//			/* test and add */
//			auto lp = p.local();
//			REQUIRE(lp->size() == 1);
//			REQUIRE(lp->at(0) == 42);
//			lp->push_back(43);
//			lp->resize(10);
//
//			/* push back to 0 */
//			p = gam::private_ptr<gam_vector<int>>(std::move(lp));
//			p.push(0);
//			break;
//		}
//		}
//	}
//}
