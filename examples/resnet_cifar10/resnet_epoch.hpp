#include <chrono>
#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Context ctx = Context::cpu(); // Use CPU for training

class ModelLogic
{
  public:
	void init()
	{

		float learning_rate = 0.01;

		// Parsing environment for config
		if (const char *sync_env = std::getenv("SYNC_EPOCHS"))
			sync_epochs_ = std::stoi(std::string(sync_env));
		else
			sync_epochs_ = 1;

		if (const char *batch_env = std::getenv("BATCH_SIZE"))
			batch_size_ = std::stoi(std::string(batch_env));
		else
			batch_size_ = 128;

		if (const char *epoch_env = std::getenv("EPOCH"))
			epoch_ = std::stoi(std::string(epoch_env));

		if (const char *learning_env = std::getenv("LEARNING_RATE"))
			learning_rate = std::stof(std::string(learning_env));
		else
			learning_rate = 0.001;
		
		std::string optimiser_name;
		if (const char *optim_env = std::getenv("OPTIMIZER"))
			optimiser_name = std::string(optim_env);
		else
			optimiser_name = "adagrad";

		if (const char *symbol_env = std::getenv("SYMBOL_JSON"))
			net = Symbol::Load(std::string(symbol_env));
		else
			net = Symbol::Load("../../symbols/resnet18_v2.json");

		std::string init_filename;
		if (const char *init_env = std::getenv("INIT_WEIGHTS"))
			init_filename = std::string(init_env);
		else
			init_filename = "../../initialized_weights/resnet18_cifar10_init_batch_" + std::to_string(batch_size_) + ".bin";

		std::string filename = "worker_" + std::to_string(FAST::rank()) + ".log";
		log_file = ofstream(filename, std::ofstream::out | std::ofstream::app);

		FAST_INFO("Batch size = " << batch_size_)
		FAST_INFO("Number of nodes = " << FAST::cardinality())

		const int image_size = 32;

		Symbol label = Symbol::Variable("label");
		net = SoftmaxOutput(net, label);


		train_iter = MXDataIter("ImageRecordIter")
						 .SetParam("path_imglist", "../../cifar10/cifar10_train.lst")
						 .SetParam("path_imgrec", "../../cifar10/cifar10_train.rec")
						 .SetParam("rand_crop", 1)
						 .SetParam("rand_mirror", 1)
						 .SetParam("data_shape", Shape(3, 32, 32))
						 .SetParam("batch_size", batch_size_)
						 .SetParam("shuffle", 1)
						 .SetParam("seed", epoch_)
						 .SetParam("preprocess_threads", 4)
						 .SetParam("num_parts", FAST::cardinality())
						 .SetParam("part_index", FAST::rank())
						 .SetParam("pad", 4)
						 .CreateDataIter();


		val_iter = MXDataIter("ImageRecordIter")
					   .SetParam("path_imglist", "../../cifar10/cifar10_val.lst")
					   .SetParam("path_imgrec", "../../cifar10/cifar10_val.rec")
					   .SetParam("rand_crop", 0)
					   .SetParam("rand_mirror", 0)
					   .SetParam("data_shape", Shape(3, 32, 32))
					   .SetParam("batch_size", batch_size_)
					   .SetParam("round_batch", 0)
					   .SetParam("preprocess_threads", 4)
					   .SetParam("pad", 4)
					   .CreateDataIter();

		FAST_DEBUG("(LOGIC): Loaded data ");

		args["data"] = NDArray(Shape(batch_size_, 3, image_size, image_size), ctx);
		args["label"] = NDArray(Shape(batch_size_), ctx);
		//Let MXNet infer shapes other parameters such as weights
		net.InferArgsMap(ctx, &args, args);

		args = mxnet::cpp::NDArray::LoadToMap(init_filename);

		opt = OptimizerRegistry::Find(optimiser_name);
		opt->SetParam("lr", learning_rate);
		opt->SetParam("wd", 1e-4);
		opt->SetParam("rescale_grad", 1.0/(batch_size_) );
     	// opt->SetParam("clip_gradient", 10);

		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();

		FAST_DEBUG("Logic initialized")
		init_time = chrono::system_clock::now();
		log_file << batch_size_ << "\t" << learning_rate << "\t" << optimiser_name << std::endl;
		log_file << "Epoch\tTime\tTraining accuracy\tTest accuracy" << std::endl;
		log_file.flush();
	}

	void run_batch()
	{
		if (max_epoch_reached)
			return;

		if (!train_iter.Next())
		{
			FAST_DEBUG("(LOGIC): next epoch");
			iter_ = 0;

			Accuracy test;
			val_iter.Reset();
			float acc;
			while (val_iter.Next())
			{
				auto data_batch = val_iter.GetDataBatch();
				data_batch.data.CopyTo(&args["data"]);
				data_batch.label.CopyTo(&args["label"]);
				// Forward pass is enough as no gradient is needed when evaluating
				exec->Forward(true);
				NDArray::WaitAll();
				test.Update(data_batch.label, exec->outputs[0]);
				NDArray::WaitAll();
				val_acc = test.Get();
			}
			sync_epochs_--;
			if (sync_epochs_ == 0) {
				max_epoch_reached = true; // Terminate
				FAST_DEBUG("(LOGIC): MAX EPOCH REACHED");
			}
			else
			{
				train_iter.Reset();
				epoch_++;
			}
			std::cerr << "=== Epoch === " << epoch_ << std::endl;
			std::cerr << "=== TRAINING ACCURACY === " << train_acc.Get() << std::endl;
			std::cerr << "=== TEST ACCURACY === " << val_acc << std::endl;

			auto toc = chrono::system_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(toc - init_time).count() / 1000.0;
			log_file << epoch_ << "\t" << duration << "\t" << train_acc.Get() << "\t\t" << val_acc << std::endl;
			log_file.flush();

			return;
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
		NDArray::WaitAll();
		// Update parameters
		for (size_t i = 0; i < arg_names.size(); ++i)
		{
			if (arg_names[i] == "data" || arg_names[i] == "label")
				continue;
			opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
		}
		FAST_INFO("Epoch = " << epoch_ << ", Samples = " << iter_ * batch_size_ << ", Training accuracy = " << train_acc.Get());
		iter_++;
	}

	void update(std::vector<mxnet::cpp::NDArray> &in)
	{
		FAST_DEBUG("(LOGIC UPDATE): updating")
		if (in.size() > 0)
		{
			NDArray::WaitAll();
			for (size_t i = 0; i < arg_names.size(); ++i)
			{
				if (arg_names[i] == "data" || arg_names[i] == "label")
					continue;
				opt->Update(i, exec->arg_arrays[i], in.at(i));
				NDArray::WaitAll();
			}
			FAST_DEBUG("(LOGIC UPDATE): updated")
		}
	}

	void finalize(bool save = false)
	{
		if (save)
		{
			std::string bestname = "best_accuracy.log";
			ofstream best_file = ofstream(bestname, std::ofstream::out | std::ofstream::app);
			best_file << epoch_ << "\t" << duration << "\t" << train_acc.Get() << "\t\t" << val_acc << std::endl;
			best_file.flush();
			mxnet::cpp::NDArray::Save("./w_" + std::to_string(FAST::rank()) + ".bin", args);
		}
	}

	Symbol net;
	std::map<std::string, NDArray> args;
	Optimizer *opt;
	Executor *exec;
	std::vector<std::string> arg_names;
	unsigned int iter_ = 0;
	unsigned int epoch_ = 0;
	unsigned int sync_epochs_ = 1;
	bool max_epoch_reached = false;
	MXDataIter train_iter = MXDataIter("ImageRecordIter");
	MXDataIter val_iter = MXDataIter("ImageRecordIter");
	Accuracy train_acc;
	float val_acc;
	int batch_size_ = 32;
	const std::string data_tag = "data";
	const std::string label_tag = "label";
	float duration;
	ofstream log_file;
	std::chrono::_V2::system_clock::time_point init_time;
};
