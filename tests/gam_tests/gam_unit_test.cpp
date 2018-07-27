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

TEST_CASE( "SPMD tensor send async", "gam,tensor" ) {
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
			FAST::future_ptr_float recv = FAST::pull_tensor_async<float>();
			auto received = recv.get();
			received->reShape(shape);
			for (uint i = 0; i < shape[0]; i++)
				for (uint j = 0; j < shape[1]; j++){
					REQUIRE(received->at(i,j) == data.at(i*shape[1]+j));
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
