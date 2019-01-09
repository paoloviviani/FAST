/*
 * example_unit_test.cpp
 *
 *  Created on: Mar 31, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include "fast.hpp"

//using namespace FAST;


TEST_CASE( "logger test", "logger" ){
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());
	int i = 100;
	FAST_DEBUG("Example of DEBUG message with integer value, " << i)
	FAST_INFO("Example of INFO message with integer value, " << i)
	FAST_ERROR("Example of ERROR message with integer value, " << i)
}
