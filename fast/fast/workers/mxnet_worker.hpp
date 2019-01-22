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
#include <map>

#include "gff.hpp"
#include "gam.hpp"

#include <ff/pipeline.hpp>
#include <ff/node.hpp>

namespace FAST {

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

typedef std::vector<mxnet::cpp::NDArray> NDAvector;

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

	InputStage(ModelLogic * logic) : logic_(logic), buffer_(nullptr), first_push_(true) {}

	void * svc(void * task) {

		auto recv_ptr = ((PublicWrapper<T> *)task)->payload.local();
		delete (PublicWrapper<T> *)task;

		FAST_DEBUG("Internal pipeline input got pointer")
		if (recv_ptr->size() == 0) {
			FAST_DEBUG("Input stage got trigger pointer")
			if (first_push_) {
				auto tmp_ptr = new NDAvector(0);
				FAST_DEBUG("Created trigger pointer for trainer stage")
				this->ff_send_out( (void*)tmp_ptr );
				FAST_DEBUG("Sent trigger pointer to trainer stage")
				first_push_ = false;
			}
			return ff::FF_GO_ON;
		}

		FAST_DEBUG("Input stage got real pointer")
		FAST::accumToNDVec( *recv_ptr, *buffer_, logic_->arg_names, "X", "label", mxnet::cpp::Context::cpu() );

		if (this->get_out_buffer()->empty()) {
			FAST_DEBUG("Input stage push gradients")
			this->ff_send_out((void *)buffer_);
			buffer_ = new NDAvector(0);
			FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "X", "label", mxnet::cpp::Context::cpu() );
		}
		return ff::FF_GO_ON;
	}

	int svc_init() {
		FAST_DEBUG("Internal pipeline input init stage")
		buffer_ = new NDAvector(0);
		FAST_DEBUG(logic_->arg_names)
		FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "X", "label", mxnet::cpp::Context::cpu() );
		FAST_DEBUG("Built NDVec");
		return 0;
	}
private:
	ModelLogic * logic_;
	NDAvector * buffer_;
	bool first_push_ = true;
};

template <typename ModelLogic, typename T >
class TrainerStage: public ff::ff_minode {
public:

	TrainerStage(ModelLogic * logic) : logic_(logic) {}

	void * svc(void * task) {
		FAST_DEBUG("Trainer stage svc")
		NDAvector * in_ptr = (NDAvector  *)task;
		NDAvector * out_grads = new NDAvector(0);
		if (this->get_channel_id() == -1 && in_ptr->size() != 0) {
			// got a pointer from the input stage
			FAST_DEBUG("Trainer stage got gradients");
			logic_->update( *in_ptr );
			logic_->run_batch( *out_grads );
			delete in_ptr;
			return (void*)out_grads;
		}

		// Got a pointer from the feedback channel
		FAST_DEBUG("Trainer stage got go on")
		logic_->run_batch(*out_grads );
		FAST_DEBUG(out_grads->size());

		return (void*)out_grads;
	}
private:
	ModelLogic * logic_;

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
			if ( ((NDAvector  *)in)->size() > 0 )
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

	OutputStage(ModelLogic * logic) : logic_(logic) {}

	void * svc(void * task) {
		FAST_DEBUG("Output stage got gradients");
		NDAvector * in_ptr = (NDAvector  *)task;
		gam_vector<T> * out = new gam_vector<T>(0);
		NDVecToVec( *in_ptr, logic_->arg_names, *out, "X", "label");
		delete in_ptr;
		FAST_DEBUG("Output stage serialized gradients of size " << out->size());

		return (void*)out;
	}

private:
	ModelLogic * logic_;
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

		FAST_DEBUG("GAM svc got pointer")

		gam_vector<T> * outvec = nullptr;
		void * outptr = (void*)outvec;
		PublicWrapper<T> * inp = new PublicWrapper<T>();
		inp->payload = in;

		FAST_DEBUG("GAM svc offloading")

		global_->offload( (void*)inp );
		FAST_DEBUG("GAM svc offloaded")
		global_->load_result( &outptr );

		FAST_DEBUG("GAM svc got results")

		FAST_DEBUG(outvec->size())
		if (outvec->size() == 0)
			return gff::eos;

		FAST_DEBUG("GAM svc preparing results")
		auto public_out = gam::public_ptr< gam_vector<T> >(outvec, [](gam_vector<T> * ptr){delete ptr;});
		FAST_DEBUG("GAM svc prepared results")

		c.emit(public_out);
		FAST_DEBUG("GAM svc emitted results")

		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		global_ = new ff::ff_pipeline(true);
		training_ = new ff::ff_pipeline();

		gam_vector<T> * ptr = new gam_vector<T>(0);

		FAST_DEBUG("Initializing model logic")
		logic_.init();
		FAST_DEBUG("Initialized model logic")


		FAST_DEBUG("Creating internal pipeline")
		global_->add_stage( new InputStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new TrainerStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new InternalAuxStage() );
		training_->wrap_around();
		global_->add_stage(training_);
		global_->add_stage( new OutputStage<ModelLogic, T>(&logic_) );

		global_->cleanup_nodes();
		training_->cleanup_nodes();

		FAST_DEBUG("Launching internal pipeline")
		global_->run();

		auto dummy_out = gam::public_ptr< gam_vector<T> >(ptr, [](gam_vector<T> * ptr){delete ptr;});

		c.emit(dummy_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		global_->offload(ff::FF_EOS);
		logic_.finalize();
		if (global_)
			delete global_;
		if (training_)
			delete training_;
	}
private:
	ff::ff_pipeline * global_;
	ff::ff_pipeline * training_;
	ModelLogic logic_;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
