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
		const std::vector<int> layers{128, 64, 32, 10};
		const int batch_size = 32;
		const float learning_rate = 0.001;

		net = mlp(layers);

		Context ctx = Context::cpu();  // Use CPU for training

		train_iter.SetParam("image", "../mnist_data/train-images-idx3-ubyte")
			  .SetParam("label", "../mnist_data/train-labels-idx1-ubyte")
			  .SetParam("batch_size", batch_size)
			  .SetParam("flat", 1)
			  .CreateDataIter();

		args["X"] = NDArray(Shape(batch_size, image_size*image_size), ctx);
		args["label"] = NDArray(Shape(batch_size), ctx);
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
			iter_ = 0;
			epoch_++;
			train_iter.Reset();
		    train_acc.Reset();
		}

		if (iter_ == 10){
			FAST_DEBUG("(LOGIC): MAX EPOCH REACHED");
			max_epoch_reached = true; // Terminate
			return;
		}

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
};