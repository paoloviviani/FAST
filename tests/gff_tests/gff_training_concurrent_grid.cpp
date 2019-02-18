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

#define SIZE      10
#define MAX_ITER      40

constexpr auto EOI_TOKEN = gff::go_on - 1;
constexpr auto TRIGGER_TOKEN = gff::go_on - 2;

template<typename T>
gam::public_ptr<T> token2public(uint64_t token) {
	return gam::public_ptr<T>(gam::GlobalPointer(token));
}
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
static auto CONSUMED_PTR = (void *)((uint64_t)ff::FF_TAG_MIN + 2);
static auto END_OF_INPUT = ff::FF_TAG_MIN;

class InputStage: public ff::ff_node {
public:

	void * svc(void * task) {
		auto recv_ptr = (FAST::gam_vector<float> *)task;

		if (recv_ptr == NEXT_ITERATION) {
			FAST_INFO("(INPUT STAGE): got trigger")
			this->ff_send_out( NEXT_ITERATION );
			return ff::FF_GO_ON;
		}

		FAST_INFO("(INPUT STAGE): got real pointer")
		for (int i = 0; i < SIZE; i++) {
			buffer->at(i) += recv_ptr->at(i);
		}
		this->ff_send_out( CONSUMED_PTR );

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

	TrainerStage(int * iter) : iter(iter) {};

	void * svc(void * task) {
		FAST_INFO("(COMPUTE STAGE): got pointer");

		if (task == CONSUMED_PTR) {
			FAST_INFO("(COMPUTE STAGE): got CONSUMED")
			return CONSUMED_PTR;
		}

		if (*iter <= MAX_ITER+1) {
			// Update internal state if received pointer
			if (task != NEXT_ITERATION) {
				FAST_INFO("(COMPUTE STAGE): got real vector")
				std::vector<float> * internal = (std::vector<float> *)task;
				for (int i = 0; i < SIZE; i++) {
					internal_state.at(i) += internal->at(i);
				}
			}
			// Compute internal iteration either if received pointer
			// Simulate some work
//			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			std::vector<float> * computed = new std::vector<float>(SIZE);
			for (int i = 0; i < SIZE; i++) {
				internal_state.at(i) += 1.;
				computed->at(i) = 0.5;
			}
			(*iter)++;
			this->ff_send_out((void*)computed);
		}

		return ff::FF_GO_ON;
	}

	int svc_init() {
		internal_state.resize(SIZE);
		for (int i = 0; i < SIZE; i++) {
			internal_state.at(i) = 1.;
		}
		return 0;
	}

private:
	std::vector<float> internal_state;
	int * iter;

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
			if (outnodes_[0]->get_out_buffer()->empty())
				ff_send_out_to(NEXT_ITERATION, 0);
			// forward the input pointer downstream
			ff_send_out_to(in, 1);
		} else {
			// send EOS to the feedback channel
			ff_send_out_to(ff::FF_EOS, 0);
		}
		return ff::FF_GO_ON;
	}

	int svc_init() {
		this->get_out_nodes(outnodes_);
		return 0;
	}
private:
	ff::svector<ff_node*> outnodes_;
};

class OutputStage: public ff::ff_node {
public:

	void * svc(void * task) {
		if (task == CONSUMED_PTR)
			return CONSUMED_PTR;
		std::vector<float> * internal = (std::vector<float> *)task;
		FAST::gam_vector<float> * computed = new FAST::gam_vector<float>(SIZE);
		for (int i = 0; i < SIZE; i++)
			computed->at(i) = internal->at(i);
		FAST_INFO("(OUTPUT STAGE): returning gradients");
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
		// Check if getting eoi
		if (in.get().address() == EOI_TOKEN) {
			assert(in.get().address() == EOI_TOKEN);
			FAST_INFO("Received EOI token")
			assert(eoi_cnt_ < (FAST::cardinality() - 1));
			if(++eoi_cnt_ == FAST::cardinality() - 1 && FAST::rank() == 0)
				return gff::eos;
		}
		else { // Run iteration
			if (iter_ < MAX_ITER ) {
				if (in.get().address() == TRIGGER_TOKEN) {
					pipe_->offload(NEXT_ITERATION);
				}
				else {
					FAST_INFO("Received pointer");
					buffer_.push( in.local() );
					pipe_->offload( (void*)(buffer_.back().get()) );
				}

				void * outptr = nullptr;
				while (true) {
					pipe_->load_result(&outptr);
					if (outptr == CONSUMED_PTR) {
						buffer_.pop();
						FAST_INFO("CONSUMED");
					}
					else {
						FAST_INFO("Running iter " << iter_);
						FAST::gam_vector<float> * out_vec = (FAST::gam_vector<float> *)outptr;
						auto out_ptr = gam::public_ptr< FAST::gam_vector<float> >(out_vec, [](FAST::gam_vector<float> * ptr){delete ptr;});
						c.emit(out_ptr);
						return gff::go_on;
					}
				}
			}
		}
		if (!eoi_out) {
			FAST_INFO("EMIT EOI");
			c.emit(token2public<FAST::gam_vector<float>>(EOI_TOKEN));
			eoi_out = true;
		}
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		pipe_ = new ff::ff_pipeline(true);
		internal_ = new ff::ff_pipeline();

		pipe_->add_stage(new InputStage());
		internal_->add_stage(new TrainerStage(&iter_));
		internal_->add_stage( new internal_out_stage() );
		internal_->wrap_around();
		pipe_->add_stage(internal_);
		pipe_->add_stage( new OutputStage() );

		pipe_->cleanup_nodes();
		internal_->cleanup_nodes();
		pipe_->run();

		c.emit(token2public<FAST::gam_vector<float>>(TRIGGER_TOKEN));
		FAST_INFO("Emitted trigger")
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		FAST_INFO("(FINALIZATION)");
		pipe_->offload( ff::FF_EOS );
		if (pipe_->wait()<0) {
			FAST_INFO("(FINALIZATION): error waiting pipe");
		}
		while (!buffer_.empty())
			buffer_.pop();
	}
private:
	std::queue < std::shared_ptr<FAST::gam_vector<float>> > buffer_;
	int eoi_cnt_ = 0;
	bool eoi_out = false;
	int iter_ = 0;
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

TEST_CASE( "gff training mockup concurrent 2D", "gam,gff,multi" ) {
	FAST_LOG_INIT
	FAST_INFO(Catch::getResultCapture().getCurrentTestName());

	const size_t grid_h = 3;
	const size_t grid_w = 3;
	size_t workers = grid_h*grid_w;

	// Row major ordering
	std::vector < std::vector< gff::NondeterminateMerge > > incoming_channels(grid_h);
	std::vector < std::vector< gff::OutBundleBroadcast<gff::NondeterminateMerge> > > outgoing_channels(grid_h);

	for (int i = 0; i < grid_h; i++) {
		for (int j = 0; j < grid_w; j++) {
			incoming_channels.at(i).emplace_back();
		}
	}

	for (unsigned int i = 0; i < grid_h; i++) {
		for (unsigned int j = 0; j < grid_w; j++) {
			outgoing_channels.at(i).emplace_back();
			// Add neighboring channels (i+1,j),(i-1,j),(i,j+1),(i,j-1) in torus topology

			unsigned int up,right,down,left;
			i == grid_h-1 ? down=0 : down=i+1;
			i == 0 ? up=grid_h-1 : up=i-1;
			j == grid_w-1 ? right=0 : right=j+1;
			j == 0 ? left=grid_w-1 : left=j-1;
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(up).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(down).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(right));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(left));
		}
	}

	for (unsigned int i = 0; i < grid_h; i++) {
		for (unsigned int j = 0; j < grid_w; j++) {
			gff::add(TrainingWorker(incoming_channels.at(i).at(j),outgoing_channels.at(i).at(j)));
		}
	}

	/* execute the network */
	gff::run();

}