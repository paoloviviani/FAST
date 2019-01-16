/*
 * worker.hpp
 *
 *  Created on: Aug 8, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_WORKERS_MXNET_WORKER_HPP_
#define FAST_FAST_WORKERS_MXNET_WORKER_HPP_

#include <array>
#include <vector>
#include "fast.hpp"

#include "gff.hpp"
#include "gam.hpp"

#include <ff/pipeline.hpp>
#include <ff/node.hpp>

/**
 * Fastflow auxiliary stuff
 */
static auto NEXT_ITERATION = (void *)((uint64_t)ff::FF_TAG_MIN + 1);
static auto END_OF_INPUT = ff::FF_TAG_MIN;

/**
 * Wrapper structs to pass objects to FF pipeline
 */
template < typename T >
struct PublicWrapper {
	gam::public_ptr < FAST::gam_vector<T> > payload;
};

template < typename T >
struct VectorWrapper {
	std::vector<T> payload;
};

/**
 * Nodes of the local pipeline
 * this pipeline executed in accelerator mode is used
 * to hide latencies related to:
 * - copy of data from public ptr to GPU/CPU memory
 * - training loop
 * - auxiliary node
 * - copy of data from GPU/CPU memory to public ptr
 */
template <typename ModelLogic, typename T >
class InputStage: public ff::ff_node {
public:

	InputStage(ModelLogic &logic) : logic_(logic), buffer_(NULL) {}

	void * svc(void * task) {

		auto recv_ptr = (PublicWrapper<T> *)task->payload.local();
		FAST::accumToNDVec( *recv_ptr, buffer_->payload, logic_.model.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		if ( /* channel empty */) {
			this->ff_send_out((void *)buffer_);
			buffer_ = new VectorWrapper<mxnet::cpp::NDArray>;
			FAST::buildNDVec( buffer_->payload, logic_.exec->grad_arrays, logic_.model.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		}
		return GO_ON;
	}

	int svc_init() {
		buffer_ = new VectorWrapper<mxnet::cpp::NDArray>;
		FAST::buildNDVec( buffer_->payload, logic_.exec->grad_arrays, logic_.model.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		return 0;
	}
private:
	ModelLogic logic_;
	VectorWrapper<mxnet::cpp::NDArray> * buffer_;
};

template <typename ModelLogic, typename T >
class TrainerStage: public ff::ff_minode {
public:

	TrainerStage(ModelLogic &logic) : logic_(logic) {}

	void * svc(void * task) {
		VectorWrapper<mxnet::cpp::NDArray> * out_grads = new VectorWrapper<mxnet::cpp::NDArray>;
		if (this->get_channel_id() == -1) {
			auto grad_arrays = (VectorWrapper<T>  *) task->payload;
			out_grads->payload = logic_.run_batch((PublicWrapper<T>  *) task);
			delete (VectorWrapper<T>  *)task;
			return (void*)out_grads;
		}
		out_grads->payload = logic_.run_batch((PublicWrapper<T>  *) task);
		return ff::FF_GO_ON;
	}
private:
	ModelLogic logic_;

	void eosnotify(ssize_t id) {
	    FAST_DEBUG("> [internal_in_stage] got EOS id=" << id << "\n");
	    if (id == 0) {
	    	// got EOS from input - forward END_OF_INPUT message
	    	FAST_DEBUG("> [internal_in_stage] sending END_OF_INPUT\n");
	    	this->ff_send_out(END_OF_INPUT);
	    } else {
	    	// got EOS from feedback - forward downstream to trigger termination
	    	FAST_DEBUG("> [internal_in_stage] sending EOS\n");
	    	this->ff_send_out(ff::FF_EOS);
	    	// got both EOSs - node will be terminated here
	    }
	}
};

class InternalAuxStage : public ff::ff_monode {
	void * svc(void * in) {
		if (in != END_OF_INPUT) {
			FAST_DEBUG("> [internal_out_stage] got " << id << "\n");
			// send a NEXT_ITERATION message to the feedback channel
			ff_send_out_to(NEXT_ITERATION, 0);
			// forward the input pointer downstream
			ff_send_out_to(in, 1);
		} else {
			FAST_DEBUG("> [internal_out_stage] got END_OF_INPUT\n");
			// send EOS to the feedback channel
			ff_send_out_to(ff::FF_EOS, 0);
		}
		return ff::FF_GO_ON;
	}
};

template < typename T >
class OutputStage: public ff::ff_node {
public:

	OutputStage(P) : {}

	void * svc(void * task) {
		std::vector<mxnet::cpp::NDArray> * grads = (std::vector<mxnet::cpp::NDArray> *) task;
		auto out_grads = out_->payload.local();
		for (auto item : *grads) {
			append(*out_grads, item);
		}
		ready = true;
		return GO_ON;
	}
private:
	PublicWrapper<T> * out_;
	bool ready;
};

namespace FAST {

/**
 * Actual worker object, to be specialised based on the specific
 * business logic (included in ModelLogic) and on the specific
 * data type (float, int, bool)
 */
template< typename ModelLogic, typename T, typename Tidx >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic(Tidx index) : training_(NULL), global_(NULL), index_(index), grad_size_(0) {}

	~MXNetWorkerLogic() {
		delete global_, training_;
	}

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		gam_vector<T> * out = new gam_vector<T>(0);

		global_.offload((void*)in->payload);
		global_.load_result(out);

		auto public_out = gam::public_ptr(out, [](gam_vector<T> * ptr){delete ptr;});

		c.emit(public_out);
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		global_ = new ff::ff_pipeline(true);
		training_ = new ff::ff_pipeline(true);

		gam_vector<T> * ptr = new gam_vector<T>(0);
		logic_.init();
		global_.add_stage( new InputStage<ModelLogic, T> (logic_) );
		training_->add_stage( new TrainerStage<ModelLogic, T> (logic_) );
		training_->add_stage( new InternalAuxStage() );
		training_->wrap_around();
		global_.add_stage( new OutputStage<T>( ) );

		global_.cleanup_nodes();
		training_->cleanup_nodes();

		global_.run();

		auto dummy_out = gam::public_ptr(ptr, [](gam_vector<T> * ptr){delete ptr;});
		c.emit(dummy_out);
	}

	void svc_end() {
		logic_.finalize();
	}
private:
	ff::ff_pipeline global_;
	ff::ff_pipeline * training_;
	ModelLogic logic_;
	Tidx index_;
	size_t grad_size_;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
