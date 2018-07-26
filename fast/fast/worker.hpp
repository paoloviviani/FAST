/*
 * worker.hpp
 *
 *  Created on: Jun 3, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_WORKER_HPP_
#define FAST_FAST_WORKER_HPP_

#include "gam.hpp"

namespace FAST {

/**
 *
 */
class Worker {
public:
	virtual ~Worker() {

	}

	void id(gam::executor_id i) {
		id__ = i;
		set_links();
	}

	gam::executor_id id() {
		return id__;
	}

	virtual void set_links() = 0;

	virtual void run() = 0;

protected:
	template<typename Logic>
	void init_(Logic &l) {
		l.svc_init();
	}

	template<typename Logic>
	void end_(Logic &l) {
		l.svc_end();
	}

protected:
	gam::executor_id id__;
};

}

#endif /* FAST_FAST_WORKER_HPP_ */
