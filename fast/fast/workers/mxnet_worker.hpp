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
#include "fast/fast.hpp"
#include "mxnet-cpp/MxNetCpp.h"

#include "gff.hpp"
#include "gam.hpp"

#include <ff/farm.hpp>
#include <ff/node.hpp>

using namespace ff;
using namespace FAST;

template < typename T >
class Loader: public ff_node {
public:
    void * svc(void * task) {
		gam::public_ptr< gam_vector<T> > incoming = (gam::public_ptr< gam_vector<T> > *) task;
		auto grads = incoming.local();
		ff_send_out((void*)grads->data());
        return GO_ON;
    }
private:
};

template <typename ModelLogic, typename T >
class Trainer: public ff_minode {
public:
    void * svc(void * task) {
    	// Update weights with received data, pass local gradients to next stage, and go on with local training
    	if (task == NULL) {
    		grads = logic.run_batch(NULL);

    	}
    	else {
    		grads = logic.run_batch((T*) task);
    	}
    	ff_send_out((void*)grads->data());
        return GO_ON;
    }
private:
    ModelLogic logic;
    std::vector<T> grads;
};

template < typename T >
class Broadcaster: public ff_node {
public:
    void * svc(void * task) {
    	T * outbound = (T*) task;
    	// Copy local gradients in tensor object, and broadcast to neighbors.
        return GO_ON;
    }
};

namespace FAST {

template< typename ModelLogic, typename T >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic() : pipe_(true /* accelerator set */) {}

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::NDOneToAll &c) {
		pipe_.offload(&in);
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		inner_pipe_.add_stage(new Trainer<ModelLogic, T>);
		inner_pipe_.wrap_around();
		pipe_.add_stage(new Loader<T>);
		pipe_.add_stage(&inner_pipe_);
		pipe_.add_stage(new Broadcaster<T>);
	}

	void svc_end() {
	}
private:
	ff_pipeline pipe_, inner_pipe_;
	array<unsigned int,2> idx_; // Index in the 2D grid of workers
};


template< typename ModelLogic, typename T >
using MXNetWorker = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< gam_vector<T> >, //
		gam::public_ptr< gam_vector<T> >, //
		MXNetWorkerLogic<ModelLogic, T> >;

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
