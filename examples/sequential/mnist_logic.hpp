#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol mlp(const std::vector<int> &layers) {
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");

	std::vector<Symbol> weights(layers.size());
	std::vector<Symbol> biases(layers.size());
	std::vector<Symbol> outputs(layers.size());

	for (size_t i = 0; i < layers.size(); ++i) {
		weights[i] = Symbol::Variable("w" + std::to_string(i));
		biases[i] = Symbol::Variable("b" + std::to_string(i));
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

		train_iter.SetParam("image", "../../mnist_data/train-images-idx3-ubyte")
			  .SetParam("label", "../../mnist_data/train-labels-idx1-ubyte")
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

		std::cout <<"Logic initialized" << std::endl;
	}


	void run_batch() {

		if (!train_iter.Next()) {
			std::cout << "(LOGIC): next epoch" << std::endl;
			iter_ = 0;
			epoch_++;
			std::cout << "=== TRAINING ACCURACY === " << train_acc.Get() << std::endl;
			train_iter.Reset();
			train_iter.Next();
		    train_acc.Reset();
		}

		if (epoch_ == 10){
			std::cout << "(LOGIC): MAX EPOCH REACHED" << std::endl;
			max_epoch_reached = true; // Terminate
			return;
		}

		// Simulate granularity
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
		if (iter_ % 100 == 0)
			std::cout << "Iter = " << iter_ << " Accuracy = " << train_acc.Get() << std::endl;
		iter_++;
	}

	void update(std::vector<mxnet::cpp::NDArray> &in) {
		if (in.size() > 0) {
			int ii = 0;
			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				opt->Update(i, exec->arg_arrays[i], in[ii]);
				ii++;
			}
			std::cout << "(LOGIC UPDATE): updated" << std::endl;
		}
	}

	void finalize() {
		  auto val_iter = MXDataIter("MNISTIter")
		      .SetParam("image", "../../data/mnist_data/t10k-images-idx3-ubyte")
		      .SetParam("label", "../../data/mnist_data/t10k-labels-idx1-ubyte")
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
		    std::cout << "=== VALIDATION ACCURACY === " << train_acc.Get() << std::endl;
	}

	Symbol net;
	std::map<std::string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	std::vector<std::string> arg_names;
	unsigned int iter_ = 0;
	unsigned int epoch_ = 0;
	bool max_epoch_reached = false;
	MXDataIter train_iter = MXDataIter("MNISTIter");
	Accuracy train_acc;
	int batch_size_ = 32;
};
