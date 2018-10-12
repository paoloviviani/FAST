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
		gam::public_ptr< gam_vector<T> > * incoming = (gam::public_ptr< gam_vector<T> > *) task;
		auto grads = incoming->local();
        return (void*)&std::move(grads);
    }
private:
};

template <typename ModelLogic, typename T >
class Trainer: public ff_minode {
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
class Broadcaster: public ff_node {
public:
    void * svc(void * task) {
    	std::vector<mxnet::cpp::NDArray> * grads = (std::vector<mxnet::cpp::NDArray> *) task;
    	gam::public_ptr< gam_vector<T> > out_grads = gam::make_public< gam_vector<T> >();
    	for (auto item : *grads) {
    		append(*out_grads, item);
    	}
        return (void*)&std::move(out_grads);
    }
};

namespace FAST {

template< typename ModelLogic, typename T >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic(unsigned int idx) : pipe_(true /* accelerator set */), idx_(idx) {}

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::NDOneToAll &c) {
		void * result = NULL;
		pipe_.offload((void*)&std::move(in));
		pipe_.load_result(&result);

		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		logic_.init;
		pipe_.add_stage( new Loader<T> );
		pipe_.add_stage( new Trainer<ModelLogic, T> (logic_) );
		pipe_.run();
	}

	void svc_end() {
		logic_.finalize();
	}
private:
	ff_pipeline pipe_;
	array<unsigned int,2> idx_; // Index in the 2D grid of workers
	ModelLogic logic_;
};


template< typename ModelLogic, typename T >
using MXNetWorker = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< gam_vector<T> >, //
		gam::public_ptr< gam_vector<T> >, //
		MXNetWorkerLogic<ModelLogic, T> >;

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
