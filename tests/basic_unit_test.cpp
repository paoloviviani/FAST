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


TEST_CASE( "logger test", "logger test" ){

	int i = 100;
	FAST_DEBUG("Debug message with integer value, " << i)
	FAST_INFO("Info message with integer value, " << i)
//	FAST_ERROR("Error message with integer value, " << i)
}
