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

TEST_CASE( "Init Tensor from NDarray", "mxnet" ){
	vector<unsigned int> shape = {50,2};
	NDArray mxnet_tensor(Shape(shape), ctx);
	NDArray::SampleUniform(0.,1.,&mxnet_tensor);

	auto tensor = FAST::Tensor(mxnet_tensor);
	REQUIRE(tensor.getShape() == mxnet_tensor.GetShape());
	REQUIRE(tensor.getSize() == 100);
	REQUIRE(mxnet_tensor == tensor.getFrameworkObject());
}
