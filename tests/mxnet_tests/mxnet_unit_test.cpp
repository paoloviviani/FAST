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

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu();  // Use CPU for training

Symbol MLP(const vector<int> layers) {
	auto input = Symbol::Variable("X");
	auto label = Symbol::Variable("label");

	vector<Symbol> weights(layers.size());
	vector<Symbol> biases(layers.size());
	vector<Symbol> outputs(layers.size());
	vector<Symbol> fc(layers.size());

	for (size_t i = 0; i < layers.size(); ++i) {
		weights[i] = Symbol::Variable("w" + to_string(i));
		biases[i] = Symbol::Variable("b" + to_string(i));
		fc[i] = FullyConnected(
				i == 0 ? input : outputs[i-1],  // data
						weights[i],
						biases[i],
						layers[i]);
		outputs[i] = (i == layers.size()-1) ? fc[i] : Activation(fc[i], ActivationActType::kRelu);
	}

	return LinearRegressionOutput("output", outputs.back(), label);
}

TEST_CASE( "Access tensor elements", "tensor" ) {
	FAST_TESTLOG(Catch::getResultCapture().getCurrentTestName())
	vector<unsigned int> shape = {6};
	vector<float> data = {11.,12.,13.,21.,22.,23.};

	FAST::Tensor<NDArray> tensor(data,shape);
	REQUIRE(tensor.getShape() == shape);
	FAST_TESTLOG("FAST tensor shape: " << tensor.getShape())
	for (uint i = 0; i < shape[0]; i++)
		REQUIRE(tensor.at(i) == data.at(i));

	shape = {2,3};
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.at(i,j) == data.at(i*shape[0]+j));
		}
}

TEST_CASE( "Init Tensor from NDarray", "mxnet, tensor" ) {
	FAST_TESTLOG(Catch::getResultCapture().getCurrentTestName())
	vector<unsigned int> shape = {10,2};
	NDArray mxnet_tensor(Shape(shape), ctx);
	NDArray::SampleUniform(0.,1.,&mxnet_tensor);

	FAST::Tensor<NDArray> tensor(mxnet_tensor);
	REQUIRE(tensor.getShape() == mxnet_tensor.GetShape());
	FAST_TESTLOG("FAST tensor shape: " << tensor.getShape())
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.at(i,j) == mxnet_tensor.At(i,j));
		}
	REQUIRE(tensor.getSize() == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>()));
}

TEST_CASE( "Init Tensor from raw pointer", "mxnet, tensor" ) {
	FAST_TESTLOG(Catch::getResultCapture().getCurrentTestName())
	vector<unsigned int> shape = {10,2};
	NDArray mxnet_tensor(Shape(shape), ctx);
	NDArray::SampleUniform(0.,1.,&mxnet_tensor);

	FAST::Tensor<NDArray> tensor(mxnet_tensor.GetData(),shape);
	REQUIRE(tensor.getShape() == mxnet_tensor.GetShape());
	FAST_TESTLOG("FAST tensor shape: " << tensor.getShape())
	for (uint i = 0; i < shape[0]; i++)
		for (uint j = 0; j < shape[1]; j++){
			REQUIRE(tensor.at(i,j) == mxnet_tensor.At(i,j));
		}
	REQUIRE(tensor.getSize() == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>()));
}
