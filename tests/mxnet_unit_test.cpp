/*
 * example_unit_test.cpp
 *
 *  Created on: May 30, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;


TEST_CASE( "mxnet unit test", "mxnet unit test tag" ){

	int input_lines = 1;
	int output_lines = 1;
	REQUIRE(input_lines == output_lines);

}
