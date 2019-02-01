/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <stdlib.h>

#include <gff.hpp>
#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Context ctx = Context::cpu();  // Use CPU for training

Symbol mlp(const std::vector<int> &layers) {
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");

	std::vector<Symbol> weights(layers.size());
	std::vector<Symbol> biases(layers.size());
	std::vector<Symbol> outputs(layers.size());

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
		const int image_size = 28;
		const std::vector<int> layers{16, 16, 10};
		batch_size_ = 32;
		const float learning_rate = 0.001;

		net = mlp(layers);

		Context ctx = Context::cpu();  // Use CPU for training

		train_iter.SetParam("image", "../mnist_data/train-images-idx3-ubyte")
			  .SetParam("label", "../mnist_data/train-labels-idx1-ubyte")
			  .SetParam("batch_size", batch_size_)
			  .SetParam("flat", 1)
			  .CreateDataIter();

		args["X"] = NDArray(Shape(batch_size_, image_size*image_size), ctx);
		args["label"] = NDArray(Shape(batch_size_), ctx);
		// Let MXNet infer shapes other parameters such as weights
		net.InferArgsMap(ctx, &args, args);

		// Initialize all parameters with uniform distribution U(-0.01, 0.01)
		auto initializer = Uniform(0.01);
		for (auto& arg : args) {
			// arg.first is parameter name, and arg.second is the value
			initializer(arg.first, &arg.second);
		}

		opt = OptimizerRegistry::Find("adam");
		opt->SetParam("lr", learning_rate);
		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();

		FAST_DEBUG("Logic initialized")
	}


	void run_batch() {
		FAST_DEBUG("(LOGIC): run batch, iteration = " << iter_);

		if (!train_iter.Next()) {
			FAST_DEBUG("(LOGIC): next epoch");
			iter_ = 0;
			epoch_++;
			FAST_INFO("=== TRAINING ACCURACY === " << train_acc.Get());
			train_iter.Reset();
		    train_acc.Reset();
		}

		if (epoch_ == 10){
			FAST_DEBUG("(LOGIC): MAX EPOCH REACHED");
			max_epoch_reached = true; // Terminate
			return;
		}

		// Simulate granularity
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		auto data_batch = train_iter.GetDataBatch();
		// Set data and label
		data_batch.data.CopyTo(&args["X"]);
		data_batch.label.CopyTo(&args["label"]);

		FAST_DEBUG("(LOGIC): running");
		// Compute gradients
		exec->Forward(true);
		exec->Backward();
		train_acc.Update(data_batch.label, exec->outputs[0]);
		// Update parameters
		for (size_t i = 0; i < arg_names.size(); ++i) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
		}
		FAST_DEBUG("(LOGIC): processed batch");
		if (iter_ % 20 == 0)
			FAST_INFO("Iter = " << iter_ << " Accuracy = " << train_acc.Get() );
		iter_++;
	}

	void update(std::vector<mxnet::cpp::NDArray> &in) {
		FAST_DEBUG("(LOGIC UPDATE): updating")
		if (in.size() > 0) {
			int ii = 0;
			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				opt->Update(i, exec->arg_arrays[i], in[ii]);
				ii++;
			}
			FAST_DEBUG("(LOGIC UPDATE): updated")
		}
	}

	void finalize() {
		  auto val_iter = MXDataIter("MNISTIter")
		      .SetParam("image", "../data/mnist_data/t10k-images-idx3-ubyte")
		      .SetParam("label", "../data/mnist_data/t10k-labels-idx1-ubyte")
			  .SetParam("batch_size", batch_size_)
			  .SetParam("flat", 1)
			  .CreateDataIter();
		  Accuracy acc;
		    val_iter.Reset();
		    while (val_iter.Next()) {
		  	auto data_batch = val_iter.GetDataBatch();
		  	data_batch.data.CopyTo(&args["X"]);
		  	data_batch.label.CopyTo(&args["label"]);
		  	// Forward pass is enough as no gradient is needed when evaluating
		  	exec->Forward(false);
		  	acc.Update(data_batch.label, exec->outputs[0]);
		    }
		    FAST_INFO("=== VALIDATION ACCURACY === " << train_acc.Get());
	}

	Symbol net;
	std::map<string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	vector<string> arg_names;
	unsigned int iter_ = 0;
	unsigned int epoch_ = 0;
	bool max_epoch_reached = false;
	MXDataIter train_iter = MXDataIter("MNISTIter");
	Accuracy train_acc;
	int batch_size_ = 32;
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

int main(int argc, char** argv) {
	FAST_LOG_INIT

	gff::NondeterminateMerge to_one, to_two;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two;

	one.add_comm(to_two);
	two.add_comm(to_one);

	gff::add(MxNetWorker(to_one,one));
	gff::add(MxNetWorker(to_two,two));

	/* execute the network */
	gff::run();

	return 0;

}
