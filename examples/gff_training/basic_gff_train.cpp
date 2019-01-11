#include <chrono>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include "fast.hpp"
#include "fast/workers/mxnet_worker.hpp"

using namespace std;
using namespace mxnet::cpp;

#define BATCH_SIZE 32
#define NEIGHBORS 2

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

class MXNetModelLogic {
public:

	gff::token_t svc(gam::public_ptr< FAST::gam_vector<float> > &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		if (local_epoch == max_epoch)
			return gff::eos;

		auto grad_local = grad_store_.local();

		if (train_iter.Next()) {
			FAST_INFO("Epoch: " << local_epoch << " iter: " << local_iter);
			auto data_batch = train_iter.GetDataBatch();
			// Set data and label
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);

			// Compute gradients
			exec->Forward(true);
			exec->Backward();
			train_acc.Update(data_batch.label, exec->outputs[0]);
			FAST_INFO("Batch processed")

			// Update parameters
			size_t offset = 0;

			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				exec->grad_arrays[i].SyncCopyToCPU(grad_local->data()+offset , exec->grad_arrays[i].Size());
				offset += exec->grad_arrays[i].Size();
				opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
			offset = 0;
			FAST_DEBUG("Before emit")
			c.emit(grad_store_);
			FAST_DEBUG("Before receive")

			if (local_epoch == 0 && local_iter < NEIGHBORS) {

				auto recv_grad = in.local();
				FAST_DEBUG("After receive")
				auto recv_ptr = recv_grad->data();

				for (size_t i = 0; i < arg_names.size(); ++i) {
					if (arg_names[i] == "X" || arg_names[i] == "label") continue;
					exec->grad_arrays[i] = NDArray(&recv_ptr[offset],Shape(exec->grad_arrays[i].GetShape()),ctx);
					offset += exec->grad_arrays[i].Size();
					FAST_DEBUG("Updating with received grads");
					opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
				}
			}
			FAST_INFO("Training accuracy: " << train_acc.Get());
			local_iter++;
		}
		else {
			local_epoch++;
			local_iter = 0;
			train_iter.Reset();
			train_acc.Reset();
		}

		return gff::go_on;

	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		const int image_size = 28;
		const vector<int> layers{32, 32, 10};
		const int batch_size = BATCH_SIZE;
		const float learning_rate = 0.001;
		const float weight_decay = 1e-4;

		train_iter.SetParam("image", "../mnist_data/train-images-idx3-ubyte")
			  .SetParam("label", "../mnist_data/train-labels-idx1-ubyte")
			  .SetParam("batch_size", batch_size)
			  .SetParam("flat", 1)
			  .CreateDataIter();
		// Fake train iterator with smaller validation data
//		train_iter.SetParam("image", "../mnist_data/t10k-images-idx3-ubyte")
//			  .SetParam("label", "../mnist_data/t10k-labels-idx1-ubyte")
//			  .SetParam("batch_size", BATCH_SIZE)
//			  .SetParam("flat", 1)
//			  .CreateDataIter();

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

		opt = OptimizerRegistry::Find("adam");
		opt->SetParam("lr", learning_rate);
		exec = net.SimpleBind(ctx, args);
		arg_names = net.ListArguments();

		grad_size = 0;
		for (size_t i = 0; i < arg_names.size(); ++i) {
			if (arg_names[i] == "X" || arg_names[i] == "label") continue;
			grad_size += exec->grad_arrays[i].Size();
		}
		FAST_DEBUG(grad_size);

		grad_store_ = gam::make_public<FAST::gam_vector<float>>(grad_size);

		c.emit(grad_store_);

//		if (train_iter.Next()) {
//					FAST_INFO("Epoch: " << local_epoch << " iter: " << local_iter);
//					auto data_batch = train_iter.GetDataBatch();
//					// Set data and label
//					data_batch.data.CopyTo(&args["X"]);
//					data_batch.label.CopyTo(&args["label"]);
//
//					// Compute gradients
//					exec->Forward(true);
//					exec->Backward();
//					train_acc.Update(data_batch.label, exec->outputs[0]);
//					FAST_INFO("Batch processed")
//					// Update parameters
//					size_t offset = 0;
//
//					auto grad_local = grad_store_.local();
//
//					for (size_t i = 0; i < arg_names.size(); ++i) {
//						if (arg_names[i] == "X" || arg_names[i] == "label") continue;
//
//						std::copy(exec->grad_arrays[i].GetData(), exec->grad_arrays[i].GetData()+exec->grad_arrays[i].Size(), //
//								grad_local->begin()+offset);
//						FAST_DEBUG(arg_names[i] << "  " << offset << "  " << exec->grad_arrays[i].Size() )
//						offset += exec->grad_arrays[i].Size();
//						opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
//						FAST_DEBUG(" ")
//					}
//					offset = 0;
//					FAST_DEBUG(" ")
//					c.emit(gam::public_ptr< FAST::gam_vector<float> >(grad_store_));
//					FAST_INFO("Training accuracy: " << train_acc.Get());
//					local_iter++;
//				}
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		auto val_iter = MXDataIter("MNISTIter")
			  .SetParam("image", "../mnist_data/t10k-images-idx3-ubyte")
			  .SetParam("label", "../mnist_data/t10k-labels-idx1-ubyte")
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
		FAST_INFO("Validation Accuracy: " << acc.Get())
	}

private:
	array<unsigned int,2> idx;
	gam::public_ptr< gam_vector<float> > grad_store_;
	MXDataIter train_iter = MXDataIter("MNISTIter");
	Symbol net;
	std::map<string, NDArray> args;
	Optimizer* opt;
	Executor * exec;
	vector<string> arg_names;
	Accuracy train_acc;
	int local_epoch = 0, local_iter = 0;
	const int max_epoch = 10;
	unsigned int grad_size;
};


typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< gam_vector<float> >, //
		gam::public_ptr< gam_vector<float> >, //
		MXNetModelLogic> MXNetWorker;


int main(int argc, char** argv) {
	FAST_LOG_INIT

	gff::NondeterminateMerge to_one, to_two, to_three;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two, three;

	one.add_comm(to_two);
	one.add_comm(to_three);
	two.add_comm(to_one);
	two.add_comm(to_three);
	three.add_comm(to_two);
	three.add_comm(to_one);

	gff::add(MXNetWorker(to_one,one));
	gff::add(MXNetWorker(to_two,two));
	gff::add(MXNetWorker(to_three,three));

	FAST_INFO("Created network")
	/* execute the network */
	gff::run();

	return 0;
}
