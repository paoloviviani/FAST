/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <catch.hpp>
#include "gff.hpp"
#include <string>
#include <cassert>
#include <cmath>
#include "fast.hpp"
#include <stdlib.h>

#define CATCH_CONFIG_MAIN

using namespace std;

#define NWORKERS    3
#define NUMBER      5

/*
 *******************************************************************************
 *
 * farm components
 *
 *******************************************************************************
 */
class WorkerLogicAllReduce {
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
	gff::token_t svc(gam::public_ptr<int> &in, gff::NDOneToAll &c) {
		FAST_DEBUG("Out use count = " << out.use_count());
		auto local_in = in.local();
		auto number = *local_in;
		in.reset();
		iter++;
		FAST_DEBUG("Received number " << number << "  " << iter << " times.");
		buffer_.push_back(number);
		if (buffer_.size() == 2) {
			sum_ = buffer_[0] + buffer_[1];
			while (out.use_count() > 1) {
				sleep(100);
			}
			return gff::eos;
		}
		return gff::go_on;
	}

	void svc_init(gff::NDOneToAll &c) {
		out = gam::make_public<int>(NUMBER);
		c.emit(gam::make_public<int>(NUMBER));
		FAST_DEBUG("Emitted number " << NUMBER);
	}

	void svc_end(gff::NDOneToAll &c) {
		REQUIRE(2*NUMBER == sum_);
	}
private:
	vector< int > buffer_;
	int iter = 0;
	int sum_ = 0;
	gam::public_ptr<int> out = nullptr;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the output channel
 * - the type of the input pointers
 * - the type of the output pointers
 * - the gff logic
 */
typedef gff::Filter<gff::NDOneToAll, gff::NDOneToAll,//
		gam::public_ptr<int>, gam::public_ptr<int>, //
		WorkerLogicAllReduce> WorkerAllReduce;
/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "gff allreduce", "gam,gff" ) {
	FAST_LOG_INIT
	gff::NDOneToAll all;

	for (unsigned i = 0; i < NWORKERS; i++)
		gff::add(WorkerAllReduce(all,all));

	/* execute the network */
	gff::run();
}
