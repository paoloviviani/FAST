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

	SyncToCPU(gam_vector<T> * out_grads, bool & ready) : out_grads(out_grads), ready(ready)  {}

    void * svc(void * task) {
    	std::vector<mxnet::cpp::NDArray> * grads = (std::vector<mxnet::cpp::NDArray> *) task;
    	for (auto item : *grads) {
    		append(*out_grads, item);
    	}
    	ready = true;
        return GO_ON;
    }
private:
    gam_vector<T> * out_grads;
    bool ready;
};

namespace FAST {

template< typename ModelLogic, typename T >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic() : pipe_(true /* accelerator set */) {}

	gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::NDOneToAll &c) {
		auto grads = in.local();
		pipe_.offload(grads->data());
		if (ready) {
			c.emit(out_grads);
		}
		ready = false;
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		gam_vector<T> * ptr = new gam_vector<T>;
		out_grads = gam::public_ptr<gam_vector<T>>(ptr);
		logic_.init();
		pipe_.add_stage( new Trainer<ModelLogic, T> (logic_) );
		pipe_.add_stage( new SyncToCPU<T>(out_grads.local().get(), ready ) );
		pipe_.run();
		c.emit(gam::make_public<gam_vector<T>>(NULL));
	}

	void svc_end() {
		logic_.finalize();
		delete out_grads.local().get();
	}
private:
	ff_pipeline pipe_;
	gam::public_ptr < gam_vector<T> > out_grads;
	ModelLogic logic_;
	bool ready = false;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
