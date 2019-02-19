#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Context ctx = Context::cpu();  // Use CPU for training

class ModelLogic {
public:
	void init() {
		batch_size_ = 128;
		const int image_size = 32;
		const float learning_rate = 0.01;
		const float weight_decay = 1e-4;

		net = Symbol::Load("../symbols/resnet50_v2.json");



		Symbol label = Symbol::Variable("label");
		net = SoftmaxOutput(net, label);

		MXRandomSeed(42);

		train_iter = MXDataIter("ImageRecordIter")
			.SetParam("path_imglist", "../cifar10/cifar10_train.lst")
			.SetParam("path_imgrec", "../cifar10/cifar10_train.rec")
			.SetParam("rand_crop", 1)
			.SetParam("rand_mirror", 1)
			.SetParam("data_shape", Shape(3, 32, 32))
			.SetParam("batch_size", batch_size_)
			.SetParam("shuffle", 1)
			.SetParam("preprocess_threads", 24)
			.SetParam("num_parts", FAST::cardinality())
			.SetParam("part_index", FAST::rank())
			.CreateDataIter();

		FAST_DEBUG("(LOGIC): Loaded data ");

		args["data"] = NDArray(Shape(batch_size_, 3, image_size, image_size), ctx);
		args["label"] = NDArray(Shape(batch_size_), ctx);
		//Let MXNet infer shapes other parameters such as weights
		net.InferArgsMap(ctx, &args, args);

		//Initialize all parameters with uniform distribution U(-0.01, 0.01)
//		auto initializer = Xavier();
//		for (auto& arg : args) {
//			//arg.first is parameter name, and arg.second is the value
//			initializer(arg.first, &arg.second);
//		}
		// Load same weights for all the workers

		args = mxnet::cpp::NDArray::LoadToMap("../initialized_weights/resnet50_cifar10_epoch0.bin");


		opt = OptimizerRegistry::Find("adam");
		opt->SetParam("lr", learning_rate);
		opt->SetParam("wd", weight_decay);

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
			NDArray::WaitAll();
			std::cout << "=== Epoch === " << epoch_ << std::endl;
			std::cout << "=== TRAINING ACCURACY === " << train_acc.Get() << std::endl;
			if (epoch_ == max_epoch_){
				FAST_DEBUG("(LOGIC): MAX EPOCH REACHED");
				max_epoch_reached = true; // Terminate
				return;
			}
			train_iter.Reset();
			train_iter.Next();
			train_acc.Reset();
			FAST_INFO("Restarted data iterator")
		}

		auto data_batch = train_iter.GetDataBatch();
		// Set data and label
		data_batch.data.CopyTo(&args["data"]);
		data_batch.label.CopyTo(&args["label"]);

		FAST_DEBUG("(LOGIC): running");
		// Compute gradients
		exec->Forward(true);
		exec->Backward();
		train_acc.Update(data_batch.label, exec->outputs[0]);
		// Update parameters
		for (size_t i = 0; i < arg_names.size(); ++i) {
			if (arg_names[i] == "data" || arg_names[i] == "label") continue;
			opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
		}
		FAST_INFO("=======================================================");
		FAST_INFO("Epoch = " << epoch_ << "Samples = " << iter_*batch_size_ );
		FAST_INFO("Accuracy = " << train_acc.Get());
		FAST_INFO("=======================================================");
		FAST_DEBUG("(LOGIC): processed batch");
		iter_++;
	}

	void update(std::vector<mxnet::cpp::NDArray> &in) {
		FAST_DEBUG("(LOGIC UPDATE): updating")
				if (in.size() > 0) {
					for (size_t i = 0; i < arg_names.size(); ++i) {
						if (arg_names[i] == "data" || arg_names[i] == "label") continue;
						opt->Update(i, exec->arg_arrays[i], in[1]);
					}
					FAST_DEBUG("(LOGIC UPDATE): updated")
				}
	}

	void finalize() {
		auto val_iter = MXDataIter("ImageRecordIter")
			.SetParam("path_imglist", "../cifar10/cifar10_val.lst")
			.SetParam("path_imgrec", "../cifar10/cifar10_val.rec")
			.SetParam("rand_crop", 0)
			.SetParam("rand_mirror", 0)
			.SetParam("data_shape", Shape(3, 32, 32))
			.SetParam("batch_size", batch_size_)
			.SetParam("round_batch", 0)
			.SetParam("preprocess_threads", 24)
			.SetParam("pad", 2)
			.CreateDataIter();

		Accuracy acc;
		val_iter.Reset();
		while (val_iter.Next()) {
			auto data_batch = val_iter.GetDataBatch();
			data_batch.data.CopyTo(&args["data"]);
			data_batch.label.CopyTo(&args["label"]);
			// Forward pass is enough as no gradient is needed when evaluating
			exec->Forward(false);
			acc.Update(data_batch.label, exec->outputs[0]);
		}
		NDArray::WaitAll();
		FAST_INFO("=== VALIDATION ACCURACY === " << train_acc.Get());
	}

	Symbol net;
	std::map<std::string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	std::vector<std::string> arg_names;
	unsigned int iter_ = 0;
	unsigned int epoch_ = 0;
	bool max_epoch_reached = false;
	MXDataIter train_iter = MXDataIter("ImageRecordIter");
	Accuracy train_acc;
	int batch_size_ = 32;
	const int max_epoch_ = 1;
	const std::string data_tag = "data";
	const std::string label_tag = "label";
};
