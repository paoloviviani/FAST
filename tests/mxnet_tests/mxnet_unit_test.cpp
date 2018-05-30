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
//using namespace FAST::mxnet;

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

TEST_CASE( "mxnet unit test", "mxnet unit test tag" ){
	const vector<int> layers{256,256,1};
	Context ctx = Context::cpu();  // Use CPU for training
	Symbol net = MLP(layers);
	int input_lines = 1;
	int output_lines = 1;
	REQUIRE(input_lines == output_lines);

}
