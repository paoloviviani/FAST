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
struct W: ff_node {

	int svc_init() {
		return 0;
	}

	void *svc(void *task){
		std::cout << "W(" << get_my_id() << ") got task " << (*(ssize_t*) task) << "\n";
		return task;
	}

	int svc_end() {
		return 0;
	}

	ModelLogic model_;

};

class E: public ff_node {
public:
	E(ff_loadbalancer *const lb):lb(lb) {}
	int svc_init() {
		eosreceived=false, numtasks=0;
		return 0;
	}
	void *svc(void *task) {
		if (lb->get_channel_id() == -1) {
			++numtasks;
			return task;
		}
		if (--numtasks == 0 && eosreceived) return EOS;
		return GO_ON;
	}
	void eosnotify(ssize_t id) {
		if (id == -1)  {
			eosreceived = true;
			if (numtasks == 0) {
				printf("BROADCAST\n");
				fflush(stdout);
				lb->broadcast_task(EOS);
			}
		}
	}
private:
	bool eosreceived;
	long numtasks;
protected:
	ff_loadbalancer *const lb;
};

namespace FAST {

template< typename ModelLogic, typename Payload >
class MXNetWorkerLogic {
public:

	MXNetWorkerLogic() : farm(true /* accelerator set */) {}

	gff::token_t svc(gam::public_ptr<Payload> &in, gff::NDOneToAll &c) {
		farm.offload(in);
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		E emitter(farm.getlb());
		farm.add_emitter(&emitter);
	}

	void svc_end() {
	}
private:
	ff_farm<> farm;
	array<unsigned int,2> idx_;
};


template< typename ModelLogic, typename Payload >
using MXNetWorker = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< Payload >, //
		gam::public_ptr< Payload >, //
		MXNetWorkerLogic<ModelLogic, Payload> >;

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
