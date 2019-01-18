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

namespace FAST {


/**
 * Wrapper structs to pass objects to FF pipeline
 */
template < typename T >
struct PublicWrapper {
	gam::public_ptr < FAST::gam_vector<T> > payload;
};

template < typename T >
struct VectorWrapper {
	FAST::gam_vector<T> payload;
};

struct ArgsVectorWrapper {
	std::vector<mxnet::cpp::NDArray> payload;
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

		auto recv_ptr = ((PublicWrapper<T> *)task)->payload.local();
		FAST::accumToNDVec( *recv_ptr, buffer_->payload, logic_.model.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		delete task;
		if (this->get_out_buffer()->empty()) {
			this->ff_send_out((void *)buffer_);
			buffer_ = new ArgsVectorWrapper();
			FAST::buildNDVec( buffer_->payload, logic_.exec->grad_arrays, logic_.net.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		}
		return GO_ON;
	}

	int svc_init() {
		buffer_ = new ArgsVectorWrapper();
		FAST::buildNDVec( buffer_->payload, logic_.exec->grad_arrays, logic_.net.ListArguments(), "X", "label", mxnet::cpp::Context::cpu() );
		return 0;
	}
private:
	ModelLogic logic_;
	ArgsVectorWrapper * buffer_;
};

template <typename ModelLogic, typename T >
class TrainerStage: public ff::ff_minode {
public:

	TrainerStage(ModelLogic &logic) : logic_(logic) {}

	void * svc(void * task) {
		ArgsVectorWrapper * out_grads = new ArgsVectorWrapper();
		if (this->get_channel_id() == -1) {
			auto grad_arrays = ((ArgsVectorWrapper  *)task)->payload;
			logic_.run_batch(grad_arrays, out_grads->payload );
			delete (ArgsVectorWrapper  *)task;
			return (void*)out_grads;
		}

		logic_.run_batch(std::vector<mxnet::cpp::NDArray>(0), out_grads->payload);
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
			FAST_DEBUG("> [internal_out_stage] got " << in << "\n");
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

template <typename ModelLogic, typename T >
class OutputStage: public ff::ff_node {
public:
	OutputStage(ModelLogic &logic) : logic_(logic) {}

	void * svc(void * task) {

		auto args_vec = ((ArgsVectorWrapper  *)task)->payload;
		gam_vector<T> * out = new gam_vector<T>(0);
		NDVecToVec(args_vec, logic_.net.ListArguments(), *out, "X", "label");
		return out;
	}

private:
	ModelLogic logic_;
};

/**
 * Actual worker object, to be specialised based on the specific
 * business logic (included in ModelLogic) and on the specific
 * data type (float, int, bool)
 */
template< typename ModelLogic, typename T>
class MXNetWorkerLogic {
public:

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		void * outptr = NULL;
		PublicWrapper<T> * inp = new PublicWrapper<T>();
		inp->payload = in;

		global_->offload( (void*)inp );
		global_->load_result( &outptr );

		auto public_out = gam::public_ptr< gam_vector<T> >((gam_vector<T> *)outptr, [](gam_vector<T> * ptr){delete ptr;});

		c.emit(public_out);
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		global_ = new ff::ff_pipeline(true);
		training_ = new ff::ff_pipeline(true);

		gam_vector<T> * ptr = new gam_vector<T>(0);
		logic_.init();
		global_->add_stage( new InputStage<ModelLogic, T>(logic_) );
		training_->add_stage( new TrainerStage<ModelLogic, T>(logic_) );
		training_->add_stage( new InternalAuxStage() );
		training_->wrap_around();
		global_->add_stage( new OutputStage<ModelLogic, T>(logic_) );

		global_->cleanup_nodes();
		training_->cleanup_nodes();

		global_->run();

		auto dummy_out = gam::public_ptr< gam_vector<T> >(ptr, [](gam_vector<T> * ptr){delete ptr;});

		c.emit(dummy_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		logic_.finalize();
	}
private:
	ff::ff_pipeline * global_;
	ff::ff_pipeline * training_;
	ModelLogic logic_;
	size_t grad_size_;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
