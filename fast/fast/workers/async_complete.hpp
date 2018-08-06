/*
 * async_complete.hpp
 *
 *  Created on: Aug 2, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_WORKERS_ASYNC_COMPLETE_HPP_
#define FAST_FAST_WORKERS_ASYNC_COMPLETE_HPP_

#include "gff.hpp"

namespace FAST {

template<typename ModelLogic>
class ACWorkerLogic {
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
	gff::token_t svc(gam::public_ptr<int> &in, gff::NondeterminateMerge &c) {
		auto local_in = in.local();
		if (*local_in < THRESHOLD)
			c.emit(gam::make_private<char>((char) std::sqrt(*local_in)));
		return gff::go_on;
	}

	void svc_init() {
	}

	void svc_end() {
	}

private:
	ModelLogic model;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the output channel
 * - the type of the input pointers
 * - the type of the output pointers
 * - the gff logic
 */

template<typename ModelLogic>
using ACWorker = gff::Filter<gff::RoundRobinSwitch, gff::NondeterminateMerge, //
		gam::public_ptr<int>, gam::public_ptr<char>, //
		ACWorkerLogic<ModelLogic>>;

} // Namespace FAST

#endif /* FAST_FAST_WORKERS_ASYNC_COMPLETE_HPP_ */
