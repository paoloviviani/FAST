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
template < typename T >
struct PublicWrapper {
	gam::public_ptr < FAST::gam_vector<T> > payload;
};

class InputStage: public ff::ff_node {
public:

	void * svc(void * task) {
		auto recv_ptr = ((PublicWrapper<float> *)task)->payload.local();
		delete (PublicWrapper<float> *)task;
		FAST_DEBUG("(INPUT STAGE): got pointer")

		if (recv_ptr->size() == 0) {
			std::vector<float> * trigger = new std::vector<float>(0);
			this->ff_send_out((void*)trigger);
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

class ComputeStage: public ff::ff_node {
public:

	void * svc(void * task) {
		std::vector<float> * internal = (std::vector<float> *)task;
		FAST::gam_vector<float> * computed = new FAST::gam_vector<float>(SIZE);

		if (internal->size() == 0) {
			FAST_DEBUG("(COMPUTE STAGE): got trigger pointer");
			for (int i = 0; i < SIZE; i++) {
				computed->at(i) = 1.;
				internal_state.at(i) = 1.;
			}
			delete internal;
			this->ff_send_out((void*)computed);
			return ff::FF_GO_ON;
		}

		for (int i = 0; i < SIZE; i++) {
			internal_state.at(i) += internal->at(i);
			computed->at(i) = 1.;
		}
		delete internal;
		this->ff_send_out((void*)computed);

		return ff::FF_GO_ON;
	}

	int svc_init() {
		internal_state.resize(SIZE);
		return 0;
	}

	void svc_end() {
		float test = MAX_ITER;
		REQUIRE(internal_state.at(0) == test);
		FAST_INFO("(svc_end): intenral_state = " << internal_state)
	}
private:
	std::vector<float> internal_state;
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

		PublicWrapper<float> * inp = new PublicWrapper<float>();
		inp->payload = in;
		FAST_INFO("Received pointer")

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
		internal_state_.resize(10);

		pipe_ = new ff::ff_pipeline(true);
		pipe_->add_stage(new InputStage());
		pipe_->add_stage(new ComputeStage());

		pipe_->cleanup_nodes();
		pipe_->run();

		FAST::gam_vector<float> * ptr = new FAST::gam_vector<float>(0);
		auto dummy_out = gam::public_ptr< FAST::gam_vector<float> >(ptr, [](FAST::gam_vector<float> * ptr){delete ptr;});
		FAST_INFO("Emitting trigger")
		c.emit(dummy_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		pipe_->offload( ff::FF_EOS );
//		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	}
private:
	vector< float > internal_state_;
	int iter = 0;
	ff::ff_pipeline * pipe_;
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

TEST_CASE( "gff training mockup", "gam,gff,multi" ) {
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
