#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol mlp(const std::vector<int> &layers)
{
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");

	std::vector<Symbol> weights(layers.size());
	std::vector<Symbol> biases(layers.size());
	std::vector<Symbol> outputs(layers.size());

	for (size_t i = 0; i < layers.size(); ++i)
	{
		weights[i] = Symbol::Variable("w" + std::to_string(i));
		biases[i] = Symbol::Variable("b" + std::to_string(i));
		Symbol fc = FullyConnected(
			i == 0 ? x : outputs[i - 1], // data
			weights[i],
			biases[i],
			layers[i]);
		outputs[i] = i == layers.size() - 1 ? fc : Activation(fc, ActivationActType::kRelu);
	}

	return SoftmaxOutput(outputs.back(), label);
}

bool isFileExists(const std::string &filename) {
  std::ifstream fhandle(filename.c_str());
  return fhandle.good();
}

class ModelLogic
{
  public:
	void init()
	{
		const int image_size = 28;
		const std::vector<int> layers{128, 64, 32, 10};
		batch_size_ = 256;
		const float learning_rate = 0.001;

		net = mlp(layers);

		Context ctx = Context::cpu(); // Use CPU for training

		train_iter.SetParam("image", "../../mnist_data/train-images-idx3-ubyte")
			.SetParam("label", "../../mnist_data/train-labels-idx1-ubyte")
			.SetParam("batch_size", batch_size_)
			.SetParam("flat", 1)
			.SetParam("shuffle", 1)
			.SetParam("num_parts", FAST::cardinality())
			.SetParam("part_index", FAST::rank())
			.CreateDataIter();

		val_iter.SetParam("image", "../../mnist_data/t10k-images-idx3-ubyte")
			.SetParam("label", "../../mnist_data/t10k-labels-idx1-ubyte")
			.SetParam("batch_size", batch_size_)
			.SetParam("flat", 1)
			.CreateDataIter();

		args["X"] = NDArray(Shape(batch_size_, image_size * image_size), ctx);
		args["label"] = NDArray(Shape(batch_size_), ctx);
		// Let MXNet infer shapes other parameters such as weights
		net.InferArgsMap(ctx, &args, args);

		// Initialize all parameters with uniform distribution U(-0.01, 0.01)
		auto initializer = Uniform(0.01);
		for (auto &arg : args)
		{
			// arg.first is parameter name, and arg.second is the value
			initializer(arg.first, &arg.second);
		}

		opt = OptimizerRegistry::Find("adam");
		opt->SetParam("lr", learning_rate);
		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();

		FAST_DEBUG("Logic initialized")
	}

	void run_batch()
	{
		FAST_DEBUG("(LOGIC): run batch, iteration = " << iter_);

		if (!train_iter.Next())
		{
			std::cout << "(LOGIC): next epoch" << std::endl;
			iter_ = 0;
			epoch_++;
			std::cout << "=== TRAINING ACCURACY === " << train_acc.Get() << std::endl;
			val_acc.Reset();
			val_iter.Reset();
			while (val_iter.Next())
			{
				auto data_batch = val_iter.GetDataBatch();
				data_batch.data.CopyTo(&args["X"]);
				data_batch.label.CopyTo(&args["label"]);
				// Forward pass is enough as no gradient is needed when evaluating
				exec->Forward(false);
				NDArray::WaitAll();
				val_acc.Update(data_batch.label, exec->outputs[0]);
			}
			std::cout << "=== TEST ACCURACY === " << train_acc.Get() << std::endl;
			if (epoch_ == 1)
			{
				FAST_INFO("(LOGIC): MAX EPOCH REACHED");
				max_epoch_reached = true; // Terminate
			}
			train_iter.Reset();
			train_iter.Next();
			train_acc.Reset();
		}

		// Simulate granularity
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		auto data_batch = train_iter.GetDataBatch();
		// Set data and label
		data_batch.data.CopyTo(&args["X"]);
		data_batch.label.CopyTo(&args["label"]);

		FAST_DEBUG("(LOGIC): running");
		// Compute gradients
		exec->Forward(true);
		exec->Backward();
		NDArray::WaitAll();
		train_acc.Update(data_batch.label, exec->outputs[0]);
		// Update parameters
		for (size_t i = 0; i < arg_names.size(); ++i)
		{
			if (arg_names[i] == "X" || arg_names[i] == "label")
				continue;
			opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
		}
		FAST_DEBUG("(LOGIC): processed batch");
		if (iter_ % 20 == 0)
		{
			FAST_INFO("=======================================================");
			FAST_INFO("Epoch = " << epoch_);
			FAST_INFO("Iter = " << iter_ << " Accuracy = " << train_acc.Get());
			FAST_INFO("=======================================================");
		}
		iter_++;
	}

	void update(std::vector<mxnet::cpp::NDArray> &in)
	{
		FAST_DEBUG("(LOGIC UPDATE): updating")
		if (in.size() > 0)
		{
			for (size_t i = 0; i < arg_names.size(); ++i)
			{
				if (arg_names[i] == "X" || arg_names[i] == "label")
					continue;
				opt->Update(i, exec->arg_arrays[i], in[i]);
				NDArray::WaitAll();
			}
			FAST_DEBUG("(LOGIC UPDATE): updated")
		}
	}

	void finalize(bool save=false)
	{
		FAST_INFO("FINALIZATION")

		Accuracy acc;
		val_iter.Reset();
		while (val_iter.Next())
		{
			auto data_batch = val_iter.GetDataBatch();
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);
			// Forward pass is enough as no gradient is needed when evaluating
			exec->Forward(false);
			NDArray::WaitAll();
			acc.Update(data_batch.label, exec->outputs[0]);
		}
		FAST_INFO("=== VALIDATION ACCURACY === " << train_acc.Get());
	}

	Symbol net;
	std::map<std::string, NDArray> args;
	Optimizer *opt;
	Executor *exec;
	std::vector<std::string> arg_names;
	unsigned int iter_ = 0;
	unsigned int epoch_ = 0;
	bool max_epoch_reached = false;
	MXDataIter train_iter = MXDataIter("MNISTIter");
	MXDataIter val_iter = MXDataIter("MNISTIter");
	Accuracy train_acc, val_acc;
	int batch_size_ = 32;
	const std::string data_tag = "X";
	const std::string label_tag = "label";
};
