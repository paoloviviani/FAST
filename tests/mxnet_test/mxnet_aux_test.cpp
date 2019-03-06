/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <catch.hpp>
#include <string>
#include <cassert>
#include <cmath>
#include <stdlib.h>

#include <gff.hpp>
#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

#define CATCH_CONFIG_MAIN

Context ctx = Context::cpu();  // Use CPU for training
typedef std::vector<mxnet::cpp::NDArray> NDAvector;

/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */


TEST_CASE( "NDArray vector to gam vector and back", "mxnet" ) {
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());
	Context ctx = Context::cpu();  // Use CPU for training

	std::vector<NDArray> grad_arrays;
	vector<string> arg_names;

	arg_names.push_back("X");
	arg_names.push_back("first");
	arg_names.push_back("second");
	arg_names.push_back("third");
	arg_names.push_back("fourth");
	arg_names.push_back("label");

	for (size_t i = 0; i < arg_names.size(); ++i) {
		grad_arrays.push_back( NDArray(Shape(8, 2), ctx) );
		grad_arrays[i] = i;
		LG << grad_arrays[i];
	}

	FAST::gam_vector<float> out;
	FAST::NDVecToVec(grad_arrays, arg_names, out, "X", "label");


	NDAvector* buffer = new NDAvector(0);
	FAST::buildNDVec( *buffer, grad_arrays, arg_names, mxnet::cpp::Context::cpu() );

	FAST::vecToNDVec(out, *buffer, arg_names, "X", "label");



	for (size_t i = 0; i < arg_names.size(); ++i) {
		for (size_t j = 0; j < buffer->at(i).Size(); j++) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			REQUIRE(grad_arrays[i].GetData()[j] == buffer->at(i).GetData()[j] );
		}
	}


}


TEST_CASE( "Accumulate NDArray from gam vector", "mxnet" ) {
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());
	Context ctx = Context::cpu();  // Use CPU for training

	std::vector<NDArray> grad_arrays;
	vector<string> arg_names;

	arg_names.push_back("X");
	arg_names.push_back("first");
	arg_names.push_back("second");
	arg_names.push_back("third");
	arg_names.push_back("fourth");
	arg_names.push_back("label");

	for (size_t i = 0; i < arg_names.size(); ++i) {
		grad_arrays.push_back( NDArray(Shape(8, 2), ctx) );
		grad_arrays[i] = i;
		LG << grad_arrays[i];
	}

	FAST::gam_vector<float> out;
	FAST::NDVecToVec(grad_arrays, arg_names, out, "X", "label");

	NDAvector* buffer = new NDAvector(0);
	FAST::buildNDVec( *buffer, grad_arrays, arg_names);

	FAST::accumToNDVec( out, *buffer, arg_names, "X", "label");
	FAST::accumToNDVec( out, *buffer, arg_names, "X", "label");
	FAST::accumToNDVec( out, *buffer, arg_names, "X", "label");
	NDArray::WaitAll();

	for (size_t i = 0; i < arg_names.size(); ++i) {
		for (size_t j = 0; j < buffer->at(i).Size(); j++) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			REQUIRE(3 * grad_arrays[i].GetData()[j] == buffer->at(i).GetData()[j] );
		}
	}


}
