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

#ifndef DATA_TAG
#define DATA_TAG X
#endif
#ifndef OUTPUT_TAG
#define OUTPUT_TAG label
#endif

namespace FAST {

/**
 * Fastflow auxiliary stuff
 */
static auto NEXT_ITERATION = (void *)((uint64_t)ff::FF_TAG_MIN + 1);
static auto CONSUMED_PTR = (void *)((uint64_t)ff::FF_TAG_MIN + 2);
static auto END_OF_INPUT = ff::FF_TAG_MIN;

constexpr auto EOI_TOKEN = gff::go_on - 1;
constexpr auto TRIGGER_TOKEN = gff::go_on - 2;

template<typename T>
gam::public_ptr<T> token2public(uint64_t token) {
	return gam::public_ptr<T>(gam::GlobalPointer(token));
}

/**
 * Wrapper structs to pass objects to FF pipeline
 */
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
		auto recv_ptr = (FAST::gam_vector<float> *)task;
		FAST_DEBUG("(INPUT STAGE): got pointer")

		if (recv_ptr == NEXT_ITERATION) {
			FAST_DEBUG("(INPUT STAGE): got trigger")
			if (first_push_){
				this->ff_send_out( NEXT_ITERATION );
				first_push_ = false;
			}
			return ff::FF_GO_ON;
		}

		FAST_DEBUG("(INPUT STAGE): got real pointer of size " << (*recv_ptr).size())
		FAST::accumToNDVec( *recv_ptr, *buffer_, logic_->arg_names, "DATA_TAG", "OUTPUT_TAG", mxnet::cpp::Context::cpu() );
		FAST_DEBUG("(INPUT STAGE): accumulated gradients")
		this->ff_send_out( CONSUMED_PTR );

		if (this->get_out_buffer()->empty()) {
			FAST_DEBUG("(INPUT STAGE): push gradients")
			this->ff_send_out((void *)buffer_);
			buffer_ = new NDAvector(0);
			FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "DATA_TAG", "OUTPUT_TAG", mxnet::cpp::Context::cpu() );
		}
		return ff::FF_GO_ON;
	}

	int svc_init() {
		FAST_DEBUG("(INPUT STAGE): init stage")
		buffer_ = new NDAvector(0);
		FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, "DATA_TAG", "OUTPUT_TAG", mxnet::cpp::Context::cpu() );
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
		if (task == CONSUMED_PTR) {
			FAST_DEBUG("(TRAINER STAGE): got CONSUMED")
			return CONSUMED_PTR;
		}

		if (!logic_->max_epoch_reached) {
			if (task != NEXT_ITERATION) {
				// got a pointer from the input stage
				FAST_DEBUG("(TRAINER STAGE): got gradients");
				NDAvector * in_ptr = (NDAvector  *)task;
				logic_->update( *in_ptr );
			}
			FAST_DEBUG("(TRAINER STAGE): updated");
			logic_->run_batch();
			FAST_DEBUG("(TRAINER STAGE): executed batch from gradients");
			return NEXT_ITERATION;
		}
		return ff::FF_GO_ON;
	}
private:
	ModelLogic * logic_;

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

template <typename ModelLogic, typename T >
class OutputStage: public ff::ff_node {
public:

	OutputStage(ModelLogic * logic) : logic_(logic){}

	void * svc(void * task) {
		if (task == CONSUMED_PTR)
			return CONSUMED_PTR;
		FAST_DEBUG("(OUTPUT STAGE): got pointer");
		gam_vector<T> * out = new gam_vector<T>();
		NDVecToVec( logic_->exec->grad_arrays, logic_->arg_names, *out, "DATA_TAG", "OUTPUT_TAG");
		FAST_DEBUG("(OUTPUT STAGE): allocated size " << out->size());
		FAST_DEBUG("(OUTPUT STAGE): serialized gradients");
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
		// Check if getting eoi
		if (in.get().address() == EOI_TOKEN) {
			assert(in.get().address() == EOI_TOKEN);
			FAST_DEBUG("Received EOI token")
			assert(eoi_cnt_ < (FAST::cardinality() - 1));
			if(++eoi_cnt_ == FAST::cardinality() - 1)
				return gff::eos;
		}
		else {
			if ( !logic_.max_epoch_reached ) {
				if (in.get().address() == TRIGGER_TOKEN) {
					pipe_->offload(NEXT_ITERATION);
				}
				else {
					FAST_DEBUG("Received pointer");
					buffer_.push( in.local() );
					pipe_->offload( (void*)(buffer_.back().get()) );
				}

				void * outptr = nullptr;
				while (true) {
					pipe_->load_result(&outptr);
					if (outptr == CONSUMED_PTR) {
						buffer_.pop();
						FAST_DEBUG("CONSUMED");
					}
					else {
						FAST::gam_vector<float> * out_vec = (FAST::gam_vector<float> *)outptr;
						auto out_ptr = gam::public_ptr< FAST::gam_vector<float> >(out_vec, [](FAST::gam_vector<float> * ptr){delete ptr;});
						c.emit(out_ptr);
						return gff::go_on;
					}
				}
			}

			if (!eoi_out) {
				FAST_DEBUG("EMIT EOI");
				c.emit(token2public<FAST::gam_vector<float>>(EOI_TOKEN));
				eoi_out = true;
			}
		}
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		pipe_ = new ff::ff_pipeline(true);
		training_ = new ff::ff_pipeline();

		gam_vector<T> * ptr = new gam_vector<T>(0);

		FAST_DEBUG("(MXNET WORKER): Initializing model logic")
		logic_.init();
		FAST_DEBUG("(MXNET WORKER): Initialized model logic")

		FAST_DEBUG("(MXNET WORKER): Creating pipeline")
		pipe_->add_stage( new InputStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new TrainerStage<ModelLogic, T>(&logic_) );
		training_->add_stage( new internal_out_stage() );
		training_->wrap_around();
		pipe_->add_stage(training_);
		pipe_->add_stage( new OutputStage<ModelLogic, T>(&logic_) );

		pipe_->cleanup_nodes();
		training_->cleanup_nodes();

		FAST_DEBUG("(MXNET WORKER): Launching pipeline")
		pipe_->run();

		FAST_DEBUG("(MXNET WORKER): Emitting trigger")
		c.emit(token2public<FAST::gam_vector<float>>(TRIGGER_TOKEN));
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		FAST_DEBUG("(FINALIZATION)");
		pipe_->offload( ff::FF_EOS );
		if (pipe_->wait()<0) {
			FAST_DEBUG("(FINALIZATION): error waiting pipe");
		}
		while (!buffer_.empty())
			buffer_.pop();
	}
private:
	ff::ff_pipeline * pipe_;
	ff::ff_pipeline * training_;
	ModelLogic logic_;
	std::queue < std::shared_ptr<FAST::gam_vector<float>> > buffer_;
	int eoi_cnt_ = 0;
	bool eoi_out = false;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
