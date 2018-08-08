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

namespace FAST {

template< typename Payload >
using public_vector = vector< gam::public_ptr< Payload > >;

template< typename Payload >
using in_buffer = vector < gam::public_ptr< public_vector<Payload> > >;

template< typename ModelLogic, typename Payload >
class MXNetWorkerLogic {
public:

	gff::token_t svc(gam::public_ptr< public_vector<Payload> > &in, gff::NDOneToAll &c) {

		model_.next()

		buffer_.push_back(in);
		if (buffer_.size() < 2)
			return gff::go_on;
		else {
			int sum = std::accumulate(buffer_.begin(), buffer_.end(), 0);
			return gff::eos;
		}
	}

	void svc_init(gff::NDOneToAll &c) {
		model_.init();
	}

	void svc_end() {
		model_.finalize();
	}
private:
	array<unsigned int,2> idx_;
	in_buffer<Payload> buffer_;
	ModelLogic model_;
};


template< typename ModelLogic, typename Payload >
using MXNetWorkerSync = gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr< public_vector<Payload> >, //
		gam::public_ptr< public_vector<Payload> >, //
		MXNetWorkerLogic<ModelLogic, Payload> >;

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */
