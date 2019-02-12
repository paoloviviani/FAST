/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <catch.hpp>
#include "gff.hpp"
#include <string>
#include <cassert>
#include <cmath>
#include "fast.hpp"
#include <stdlib.h>

#define CATCH_CONFIG_MAIN

using namespace std;

#define NWORKERS    2
#define SIZE      10
#define MAX_ITER      100

/*
 *******************************************************************************
 *
 * Pipeline components
 *
 *******************************************************************************
 */

/**
 * Fastflow auxiliary stuff
 */
static auto NEXT_ITERATION = (void *)((uint64_t)ff::FF_TAG_MIN + 1);
static auto END_OF_INPUT = ff::FF_TAG_MIN;

template < typename T >
struct PublicWrapper {
	gam::public_ptr < FAST::gam_vector<T> > payload;
};

class InputStage: public ff::ff_node {
public:

	void * svc(void * task) {
		FAST_DEBUG("(INPUT STAGE): got pointer")
		auto recv_ptr = ((PublicWrapper<float> *)task)->payload.local();
		delete (PublicWrapper<float> *)task;

		if (recv_ptr->size() == 0) {
			FAST_DEBUG("(INPUT STAGE): got trigger")
			this->ff_send_out(NEXT_ITERATION);
			FAST_DEBUG("(INPUT STAGE): returned trigger")
			return ff::FF_GO_ON;
		}

		for (int i = 0; i < SIZE; i++) {
			buffer->at(i) += recv_ptr->at(i);
		}

		if (this->get_out_buffer()->empty()) {
			this->ff_send_out((void*)buffer);
			buffer = new std::vector<float>(SIZE);
			for (int i = 0; i < SIZE; i++)
				buffer->at(i) = 0.;
		}
		return ff::FF_GO_ON;
	}

	int svc_init() {
		buffer = new std::vector<float>(SIZE);
		for (int i = 0; i < SIZE; i++)
			buffer->at(i) = 0.;
		return 0;
	}
private:
	std::vector<float> * buffer;
};

class TrainerStage: public ff::ff_minode {
public:

	void * svc(void * task) {
		FAST_DEBUG("(COMPUTE STAGE): got pointer")
		std::vector<float> * computed = new std::vector<float>(SIZE);

		// Update internal state if received pointer
		if (task != NEXT_ITERATION) {
			FAST_DEBUG("(COMPUTE STAGE): got real vector")
			std::vector<float> * internal = (std::vector<float> *)task;
			for (int i = 0; i < SIZE; i++) {
				internal_state.at(i) += internal->at(i);
			}
			delete internal;
		}
		FAST_DEBUG("(COMPUTE STAGE): running internal iter")
		// Compute internal iteration either if received pointer or feedback or trigger
		for (int i = 0; i < SIZE; i++) {
			internal_state.at(i) += 1.;
			computed->at(i) = 0.5;
		}
		iter++;
		this->ff_send_out((void*)computed);

		return ff::FF_GO_ON;
	}

	int svc_init() {
		internal_state.resize(SIZE);
		for (int i = 0; i < SIZE; i++) {
			internal_state.at(i) = 1.;
		}
		iter = 0;
		return 0;
	}

private:
	std::vector<float> internal_state;
	int iter;

	void eosnotify(ssize_t id) {
	    if (id == 0) {
	    	// got EOS from input - forward END_OF_INPUT message
	    	this->ff_send_out(END_OF_INPUT);
	    } else {
	    	// got EOS from feedback - forward downstream to trigger termination
	    	this->ff_send_out(ff::FF_EOS);
	    	// got both EOSs - node will be terminated here
	    }
	}
};

class internal_out_stage : public ff::ff_monode {
  void *svc(void *in) {
    if (in != END_OF_INPUT) {
      // send a NEXT_ITERATION message to the feedback channel
      ff_send_out_to(NEXT_ITERATION, 0);
      // forward the input pointer downstream
      ff_send_out_to(in, 1);
    } else {
      // send EOS to the feedback channel
      ff_send_out_to(ff::FF_EOS, 0);
    }
    return ff::FF_GO_ON;
  }
};

class OutputStage: public ff::ff_node {
public:

	void * svc(void * task) {
		FAST_DEBUG("(OUTPUT STAGE)");
		std::vector<float> * internal = (std::vector<float> *)task;
		FAST::gam_vector<float> * computed = new FAST::gam_vector<float>(SIZE);
		for (int i = 0; i < SIZE; i++)
			computed->at(i) = internal->at(i);
		FAST_DEBUG("(OUTPUT STAGE): returning gradients");
		delete internal;
		return (void*)computed;
	}

private:
};

/*
 *******************************************************************************
 *
 * GFF components
 *
 *******************************************************************************
 */
class WorkerLogic {
public:
	/**
	 * The svc function is called upon each incoming pointer from upstream.
	 * Pointers are sent downstream by calling the emit() function on the
	 * output channel, that is passed as input argument.
	 *
	 * @param in is the input pointer
	 * @param c is the output channel (could be a template for simplicity)
	 * @return a gff token
	 */
	gff::token_t svc(gam::public_ptr<FAST::gam_vector<float>> &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		FAST_DEBUG("Received pointer")
		PublicWrapper<float> * inp = new PublicWrapper<float>();
		inp->payload = in;

		FAST_DEBUG("Offloading pointer")
		pipe_->offload( (void*)inp );

		void * outptr = nullptr;
		pipe_->load_result(&outptr);

		iter++;
		if (iter == MAX_ITER && FAST::rank() == 0)
			return gff::eos;

		FAST::gam_vector<float> * out_vec = (FAST::gam_vector<float> *)outptr;

		auto out_ptr = gam::public_ptr< FAST::gam_vector<float> >(out_vec, [](FAST::gam_vector<float> * ptr){delete ptr;});
		c.emit(out_ptr);
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		pipe_ = new ff::ff_pipeline(true);
		internal_ = new ff::ff_pipeline();

		pipe_->add_stage(new InputStage());
		internal_->add_stage(new TrainerStage());
		internal_->add_stage( new internal_out_stage() );
		internal_->wrap_around();
		pipe_->add_stage(internal_);
		pipe_->add_stage( new OutputStage() );

		pipe_->cleanup_nodes();
		internal_->cleanup_nodes();
		pipe_->run();

		FAST::gam_vector<float> * ptr = new FAST::gam_vector<float>(0);
		auto dummy_out = gam::public_ptr< FAST::gam_vector<float> >(ptr, [](FAST::gam_vector<float> * ptr){delete ptr;});
		c.emit(dummy_out);
		FAST_DEBUG("Emitted trigger")
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		pipe_->offload( ff::FF_EOS );
        if (pipe_->wait()<0) {
        	FAST_INFO("(FINALIZATION): error wating pipe");
        }
//		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	}
private:
	int iter = 0;
	ff::ff_pipeline * pipe_;
	ff::ff_pipeline * internal_;
};


typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, gam::public_ptr< FAST::gam_vector<float> >, //
		WorkerLogic > TrainingWorker;
/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "gff training mockup concurrent", "gam,gff,multi" ) {
	FAST_LOG_INIT
	FAST_INFO(Catch::getResultCapture().getCurrentTestName());

	gff::NondeterminateMerge to_one, to_two;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two;

	one.add_comm(to_two);
	two.add_comm(to_one);

	gff::add(TrainingWorker(to_one,one));
	gff::add(TrainingWorker(to_two,two));

	/* execute the network */
	gff::run();

}
