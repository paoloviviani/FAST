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

using namespace ff;
using namespace FAST;


/**
 * Wrapper struct to pass public pointers to FF pipeline
 */
template < typename T >
struct PublicWrapper {
	gam::public_ptr < gam_vector<T> > payload;
};

/**
 * Nodes of the local pipeline
 * this pipeline executed in accelerator mode is used
 * to hide latencies related to:
 * - copy of data from source
 * - copy of data to GPU memory
 * - copy of data from GPU memory
 */
template <typename ModelLogic, typename T >
class Trainer: public ff_node {
public:

	Trainer(ModelLogic &logic) : logic(logic) {}

	void * svc(void * task) {
    	// Update weights with received data, pass local gradients to next stage, and go on with local training
    	std::vector<mxnet::cpp::NDArray> * grads = logic.run_batch((T*) task);
    	return (void*)grads;
    }
private:
    ModelLogic logic;
};

template < typename T >
class SyncToCPU: public ff_node {
public:

	SyncToCPU(PublicWrapper<T> * out, bool & ready) : out_(out), ready(ready)  {}

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
template< typename ModelLogic, typename T >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic() : pipe_(true /* accelerator set */), ready_(false), in_(NULL), out_(NULL) {}

	~MXNetWorkerLogic() {
		delete in_;
		delete out_;
	}

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::NDOneToAll &c) {
		in_->payload = in;
		pipe_.offload((void*)in_);
		if (ready_) {
			c.emit(out_->payload);
		}
		ready_ = false;
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		in_ = new PublicWrapper<T>();
		out_ = new PublicWrapper<T>();
		gam_vector<T> * ptr = new gam_vector<T>;
		logic_.init();
		pipe_.add_stage( new Trainer<ModelLogic, T> (logic_) );
		pipe_.add_stage( new SyncToCPU<T>(out_, ready_ ) );
		pipe_.run();
		c.emit(gam::make_public<gam_vector<T>>(NULL));
	}

	void svc_end() {
		logic_.finalize();
	}
private:
	ff_pipeline pipe_;
	ModelLogic logic_;
	bool ready_;
	PublicWrapper<T> * in_;
	PublicWrapper<T> * out_;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
