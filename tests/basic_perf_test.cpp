/*
 * example_unit_test.cpp
 *
 *  Created on: Mar 31, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include <chrono>

//using namespace fast;
using namespace std;

TEST_CASE( "benchmarked", "[benchmark]" ) {

	static long count = 0;
	const long max_iter = 1e9;
	auto start = chrono::steady_clock::now();
	BENCHMARK( "Benchmark" ) {
	for (int i = 0; i < max_iter; i++)
		count++;
	}
	auto end = chrono::steady_clock::now();
	auto timing = chrono::duration <double,milli>(end - start).count();

	CHECK(count == max_iter);
}
