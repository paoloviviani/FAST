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

Symbol mlp(const vector<int> &layers) {
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");

	vector<Symbol> weights(layers.size());
	vector<Symbol> biases(layers.size());
	vector<Symbol> outputs(layers.size());

	for (size_t i = 0; i < layers.size(); ++i) {
		weights[i] = Symbol::Variable("w" + to_string(i));
		biases[i] = Symbol::Variable("b" + to_string(i));
		Symbol fc = FullyConnected(
				i == 0? x : outputs[i-1],  // data
						weights[i],
						biases[i],
						layers[i]);
		outputs[i] = i == layers.size()-1 ? fc : Activation(fc, ActivationActType::kRelu);
	}

	return SoftmaxOutput(outputs.back(), label);
}

class ModelLogic {
public:
	void init() {
		net = mlp({2,2,1});

		Context ctx = Context::cpu();  // Use CPU for training

		args["X"] = NDArray(Shape(1, 2), ctx);
		args["label"] = NDArray(Shape(1), ctx);
		// Let MXNet infer shapes other parameters such as weights
		net.InferArgsMap(ctx, &args, args);

		// Initialize all parameters with uniform distribution U(-0.01, 0.01)
		auto initializer = Uniform(0.01);
		for (auto& arg : args) {
			// arg.first is parameter name, and arg.second is the value
			initializer(arg.first, &arg.second);
		}

		opt = OptimizerRegistry::Find("adam");
		opt->SetParam("lr", 0.001);
		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();
		//Dummy init grads
		args["X"] = 1.;
		args["label"] = 1.;
		exec->Forward(true);
		exec->Backward();
		FAST_DEBUG("Logic initialized")
	}


	void run_batch(bool * out) {
		FAST_DEBUG("LOGIC: run batch")
		FAST_DEBUG(arg_names)
		for (size_t i = 0; i < arg_names.size(); ++i) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			exec->grad_arrays[i] += 1.;
			if (iter_ == 10)
				*out = true; // Terminate
			iter_++;
		}
	}

	void update(std::vector<mxnet::cpp::NDArray> &in) {
		if (in.size() > 0) {
			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				exec->grad_arrays[i] += in[i];
			}
		}
	}

	void finalize() {

	}

	Symbol net;
	std::map<string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	vector<string> arg_names;
	size_t iter_ = 0;
};

typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, gam::public_ptr< FAST::gam_vector<float> >, //
		FAST::MXNetWorkerLogic<ModelLogic, float> > MxNetWorker;

/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "Tensor passing basic", "gam,gff,multi,mxnet" ) {
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());

	gff::NondeterminateMerge to_one, to_two;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two;

	one.add_comm(to_two);
	two.add_comm(to_one);

	gff::add(MxNetWorker(to_one,one));
	gff::add(MxNetWorker(to_two,two));

	/* execute the network */
	gff::run();

}
