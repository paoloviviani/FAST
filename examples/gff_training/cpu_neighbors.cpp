#include <chrono>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include "fast.hpp"
#include "gam.hpp"
#include "gff.hpp"

using namespace std;
using namespace mxnet::cpp;

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


		buffer_.push_back(in);
		if (buffer_.size() < 2)
			return gff::go_on;
		else {
			int sum = std::accumulate(buffer_.begin(), buffer_.end(), 0);
			return gff::eos;
		}
	}

	void svc_init(gff::NDOneToAll &c) {
		model_.init();
	}

	void svc_end() {
		model_.finalize();
	}
private:
	array<unsigned int,2> idx;
	MXDataIter train_iter;
	Symbol net;
	std::map<string, NDArray> args;
	Executor * exec;
};


using MXNetWorkerSync = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< FAST::gam_vector<float> >, //
		gam::public_ptr< FAST::gam_vector<float> >, //
		MXNetWorkerLogic >;

void main(int argc, char** argv) {

}
