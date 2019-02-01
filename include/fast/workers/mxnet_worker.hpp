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
#include <chrono>

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
		FAST_DEBUG("(INPUT STAGE): started svc")
		auto recv_ptr = ((PublicWrapper<T> *)task)->payload.local();
		delete (PublicWrapper<T> *)task;
		FAST_DEBUG("(INPUT STAGE): got pointer")

		if (recv_ptr->size() == 0) {
			FAST_DEBUG("(INPUT STAGE): got trigger pointer")
			if (first_push_) {
				auto tmp_ptr = new NDAvector(0);
				FAST_DEBUG("Created trigger pointer for trainer stage")
				this->ff_send_out( (void*)tmp_ptr );
				FAST_DEBUG("Sent trigger pointer to trainer stage")
				first_push_ = false;
			}
			return ff::FF_GO_ON;
		}

		FAST_DEBUG("(INPUT STAGE): got real pointer of size " << (*recv_ptr).size())
		FAST::accumToNDVec( *recv_ptr, *buffer_, mxnet::cpp::Context::cpu() );
		FAST_DEBUG("(INPUT STAGE): accumulated gradients")

		if (this->get_out_buffer()->empty()) {
			FAST_DEBUG("(INPUT STAGE): push gradients")
			this->ff_send_out((void *)buffer_);
			buffer_ = new NDAvector(0);
			FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "X", "label", mxnet::cpp::Context::cpu() );
		}
		return ff::FF_GO_ON;
	}

	int svc_init() {
		FAST_DEBUG("(INPUT STAGE): init stage")
		buffer_ = new NDAvector(0);
		FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "X", "label", mxnet::cpp::Context::cpu() );
		FAST_DEBUG("(INPUT STAGE): Built NDVec");
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
		FAST_DEBUG("(TRAINER STAGE): started svc");
		bool * trigger = new bool(true);
		if (this->get_channel_id() == -1) {
			// got a pointer from the input stage
			NDAvector * in_ptr = (NDAvector  *)task;
			FAST_DEBUG("(TRAINER STAGE): got gradients");
			if (in_ptr->size() != 0)
				logic_->update( *in_ptr );
			FAST_DEBUG("(TRAINER STAGE): updated");
			logic_->run_batch();
			delete in_ptr;
			FAST_DEBUG("(TRAINER STAGE): executed batch from gradients");
			return (void*)trigger;
		}


		// Got a pointer from the feedback channel
		FAST_DEBUG("(TRAINER STAGE): got feedback go on");
		logic_->run_batch();
		FAST_DEBUG("(TRAINER STAGE): executed batch from feedback");
		return (void*)trigger;
	}
private:
	ModelLogic * logic_;

	void eosnotify(ssize_t id) {
	    FAST_DEBUG("(TRAINER STAGE): > [internal_in_stage] got EOS id=" << id);
	    if (id == 0) {
	    	// got EOS from input - forward END_OF_INPUT message
	    	FAST_DEBUG("(TRAINER STAGE): > [internal_in_stage] sending END_OF_INPUT");
	    	this->ff_send_out(END_OF_INPUT);
	    } else {
	    	// got EOS from feedback - forward downstream to trigger termination
	    	FAST_DEBUG("(TRAINER STAGE): > [internal_in_stage] sending EOS");
	    	this->ff_send_out(ff::FF_EOS);
	    	// got both EOSs - node will be terminated here
	    }
	}
};

template <typename ModelLogic>
class InternalAuxStage : public ff::ff_monode {
public:
	InternalAuxStage(ModelLogic * logic) : logic_(logic) {}

	void * svc(void * in) {
		if (in != END_OF_INPUT) {
			// send a NEXT_ITERATION message to the feedback channel
			FAST_DEBUG("(AUX STAGE): got pointer");
			if (logic_->max_epoch_reached == false) {
				ff_send_out_to(NEXT_ITERATION, 0);
				FAST_DEBUG("(AUX STAGE): forwarded feedback");
			}
			// forward the input pointer downstream
			ff_send_out_to(in, 1);
			FAST_DEBUG("(AUX STAGE): forwarded pointer");
		} else {
			// send EOS to the feedback channel
			ff_send_out_to(ff::FF_EOS, 0);
			FAST_DEBUG("(AUX STAGE): sent eos");
		}
		return ff::FF_GO_ON;
	}
private:
	ModelLogic * logic_;
};

template <typename ModelLogic, typename T >
class OutputStage: public ff::ff_node {
public:

	OutputStage(ModelLogic * logic, gff::OutBundleBroadcast<gff::NondeterminateMerge> & c) : logic_(logic), c_(c) {}

	void * svc(void * task) {
		FAST_DEBUG("(OUTPUT STAGE): got pointer");
		delete (bool *)task;
		gam_vector<T> * out = new gam_vector<T>();
		NDVecToVec( logic_->exec->grad_arrays, logic_->arg_names, *out, "X", "label");
		FAST_DEBUG("(OUTPUT STAGE): allocated size " << out->size());
		FAST_DEBUG("(OUTPUT STAGE): serialized gradients");

//		gam_vector<T> * out = (gam_vector<T> *)outptr;
//		auto public_out = gam::public_ptr< gam_vector<T> >(out, [](gam_vector<T> * ptr){delete ptr;});
//		FAST_DEBUG("(OUTPUT STAGE): prepared results")
//		c_.emit(public_out);
//		FAST_DEBUG("(OUTPUT STAGE): emitted results")

		return (void*)out;
	}

private:
	ModelLogic * logic_;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> c_;
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

		FAST_DEBUG("(MXNET WORKER): svc got pointer")

		void * outptr = nullptr;
		PublicWrapper<T> * inp = new PublicWrapper<T>();
		inp->payload = in;

		global_->offload( (void*)inp );
		FAST_DEBUG("(MXNET WORKER): svc offloaded")

		if (logic_.max_epoch_reached == true){
			FAST_DEBUG(" MAX REACHED, terminating ========== ")	
			return gff::eos;
		}

		
		global_->load_result(&outptr);
		FAST_DEBUG("(MXNET WORKER): loaded results")
//		delete (bool*)outptr;
		gam_vector<T> * out = (gam_vector<T> *)outptr;
		auto public_out = gam::public_ptr< gam_vector<T> >(out, [](gam_vector<T> * ptr){delete ptr;});
		FAST_DEBUG("(MXNET WORKER): prepared results")
		c.emit(public_out);
		FAST_DEBUG("(MXNET WORKER): emitted results")

		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		global_ = new ff::ff_pipeline(true);
		training_ = new ff::ff_pipeline();

		gam_vector<T> * ptr = new gam_vector<T>(0);

		FAST_DEBUG("(MXNET WORKER): Initializing model logic")
		logic_.init();
		FAST_DEBUG("(MXNET WORKER): Initialized model logic")

		FAST_DEBUG("(MXNET WORKER): Creating pipeline")
		global_->add_stage( new InputStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new TrainerStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new InternalAuxStage<ModelLogic>(&logic_) );
		training_->wrap_around();
		global_->add_stage(training_);
		global_->add_stage( new OutputStage<ModelLogic, T>(&logic_, c) );

		global_->cleanup_nodes();
		training_->cleanup_nodes();

		FAST_DEBUG("(MXNET WORKER): Launching pipeline")
		global_->run();

		auto dummy_out = gam::public_ptr< gam_vector<T> >(ptr, [](gam_vector<T> * ptr){delete ptr;});
		FAST_DEBUG("(MXNET WORKER): Emitting trigger")
		c.emit(dummy_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		FAST_DEBUG("(MXNET WORKER): Offloading EOS task")
		global_->offload((void *)ff::FF_EOS);
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));

		// wait EOS
		if (training_->wait_freezing()<0) {
			FAST_DEBUG("(MXNET FINALIZATION): freezing error");
		}
//        // join all threads
//        if (training_->wait()<0) {
//        	FAST_DEBUG("(MXNET FINALIZATION): error waiting pipe");
//        }
//        if (global_->wait_freezing()<0) {
//        	FAST_DEBUG("(MXNET FINALIZATION): freezing error");
//        }
//        // join all threads
//        if (global_->wait()<0) {
//        	FAST_DEBUG("(MXNET FINALIZATION): error wating pipe");
//        }
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
