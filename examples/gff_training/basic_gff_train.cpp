#include <chrono>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include "fast.hpp"
#include "gam.hpp"
#include "gff.hpp"

using namespace std;
using namespace mxnet::cpp;

#define BATCH_SIZE 128

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

class MXNetWorkerLogic {
public:

	gff::token_t svc(gam::public_ptr< FAST::gam_vector<float> > &in, gff::NDOneToAll &c) {
		if (local_epoch == max_epoch)
			return gff::eos;

		if (train_iter.Next()) {
			auto data_batch = train_iter.GetDataBatch();
			// Set data and label
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);

			// Compute gradients
			exec->Forward(true);
			exec->Backward();
			train_acc.Update(data_batch.label, exec->outputs[0]);
			// Update parameters
			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
		}
		else {
			local_epoch++;
			train_iter.Reset();
			train_acc.Reset();
		}

		return gff::go_on;

	}

	void svc_init(gff::NDOneToAll &c) {
		const int image_size = 28;
		const vector<int> layers{128, 64, 10};
		const int batch_size = BATCH_SIZE;
		const float learning_rate = 0.001;
		const float weight_decay = 1e-4;

		train_iter = MXDataIter("MNISTIter")
				  .SetParam("image", "../data/mnist_data/train-images-idx3-ubyte")
				  .SetParam("label", "../data/mnist_data/train-labels-idx1-ubyte")
				  .SetParam("batch_size", batch_size)
				  .SetParam("flat", 1)
				  .CreateDataIter();

		net = mlp(layers);

		Context ctx = Context::cpu();  // Use CPU for training

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

		opt = OptimizerRegistry::Find("sgd");
		opt->SetParam("rescale_grad", 1.0/batch_size)
			->SetParam("lr", learning_rate);
		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();

		grad_size = 0;
		for (size_t i = 0; i < arg_names.size(); ++i) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			grad_size += exec->grad_arrays[i].Size();
		}

	}

	void svc_end() {
		auto val_iter = MXDataIter("MNISTIter")
			  .SetParam("image", "../data/mnist_data/t10k-images-idx3-ubyte")
			  .SetParam("label", "../data/mnist_data/t10k-labels-idx1-ubyte")
			  .SetParam("batch_size", BATCH_SIZE)
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
	}

private:
	array<unsigned int,2> idx;
	MXDataIter train_iter;
	Symbol net;
	std::map<string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	vector<string> arg_names;
	Accuracy train_acc;
	unsigned int local_epoch = 0, local_iter = 0, samples = 0;
	const int max_epoch = 2;
	gam::public_ptr< FAST::gam_tensor<float> > grad_store;
	unsigned int grad_size;
};


using MXNetWorker = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< FAST::gam_vector<float> >, //
		gam::public_ptr< FAST::gam_vector<float> >, //
		MXNetWorkerLogic >;

void main(int argc, char** argv) {

}
