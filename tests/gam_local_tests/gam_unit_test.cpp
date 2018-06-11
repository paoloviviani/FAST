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
	FAST_DEBUG("Private pointer value: " << *p.local() );
	REQUIRE(*p.local() == 42);
}

TEST_CASE( "Gam private vector", "gam" ) {
	gam::private_ptr<vector<float>> data;
	data = gam::make_private<vector<float>>(6);
	vector<float> buf = {11.,12.,13.,21.,22.,23.};
	*data.local() = buf;
	FAST_DEBUG(*data.local());
}
