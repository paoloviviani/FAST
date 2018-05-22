/*
 * example_unit_test.cpp
 *
 *  Created on: Mar 31, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>

//using namespace fast;


TEST_CASE( "basic perf test", "basic perf test tag" ){

	int input_lines = 1;
	int output_lines = 1;
	REQUIRE(input_lines == output_lines);

}
