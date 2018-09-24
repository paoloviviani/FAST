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
#include "mxnet-cpp/MxNetCpp.h"

#include "gff.hpp"
#include "gam.hpp"
#include "gam_vector.hpp"

#include <ff/farm.hpp>
#include <ff/node.hpp>

using namespace ff;

template <typename ModelLogic>
class Ingestor: public ff_node {
public:
    void * svc(void * task) {
    	// copy data to local memory and pass on to next stage
        return task;
    }
};

template <typename ModelLogic>
class Trainer: public ff_node {
public:
    void * svc(void * task) {
        // Update weights with received data, pass local gradients to next stage, and go on with local training
        return task;
    }
};

template <typename ModelLogic>
class Broadcaster: public ff_node {
public:
    void * svc(void * task) {
    	// Copy local gradients in tensor object, and broadcast to neighbors.
        return task;
    }
};

namespace FAST {

template< typename ModelLogic, typename Payload >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic() : farm(true /* accelerator set */) {}

	gff::token_t svc(gam::public_ptr<Payload> &in, gff::NDOneToAll &c) {
		pipe_.offload(in);
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		pipe_ = ff_pipeline(true /* accelerator flag */);
		pipe_.add_stage(new Ingestor);
		pipe_.add_stage(new Trainer);
		pipe_.add_stage(new Broadcaster);
	}

	void svc_end() {
	}
private:
	ff_pipeline pipe_;
	array<unsigned int,2> idx_;
};


template< typename ModelLogic, typename Payload >
using MXNetWorker = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< Payload >, //
		gam::public_ptr< Payload >, //
		MXNetWorkerLogic<ModelLogic, Payload> >;

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
