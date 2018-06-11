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

#define FAST_TESTLOG(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_error() << \
		"TEST: " << Catch::getResultCapture().getCurrentTestName() << " -- " << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}

#define FAST_TESTNAME FAST_TESTLOG("")

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu();  // Use CPU for training

TEST_CASE( "Init Tensor from raw pointer", "tensor" ) {
	vector<unsigned int> shape = {6};
	vector<float> data = {11.,12.,13.,21.,22.,23.};
	FAST_ERROR("DEBUG")
	FAST::Tensor<NDArray> tensor(data.data(),shape);
	REQUIRE(tensor.getShape() == shape);
	REQUIRE(std::equal(std::begin(data), std::end(data), tensor.getRawPtr()));
}


TEST_CASE( "Access tensor elements", "tensor" ) {
	vector<unsigned int> shape = {6};
	vector<float> data = {11.,12.,13.,21.,22.,23.};

	FAST::Tensor<NDArray> tensor(data.data(),shape);
	REQUIRE(tensor.getShape() == shape);
	for (uint i = 0; i < shape[0]; i++)
		REQUIRE(tensor.at(i) == data.at(i));

	shape = {2,3};
	tensor.setShape(shape);
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.getRawPtr()[i*shape[1]+j] == data.at(i*shape[1]+j));
			REQUIRE(tensor.at(i,j) == data.at(i*shape[1]+j));
		}
}

TEST_CASE( "Init Tensor from NDarray", "mxnet, tensor" ) {
	vector<unsigned int> shape = {10,2};
	NDArray mxnet_tensor(Shape(shape), ctx);
	NDArray::SampleUniform(0.,1.,&mxnet_tensor);

	FAST::Tensor<NDArray> tensor(mxnet_tensor);
	REQUIRE(tensor.getShape() == mxnet_tensor.GetShape());
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.at(i,j) == mxnet_tensor.At(i,j));
		}
	REQUIRE(tensor.getSize() == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>()));
}
