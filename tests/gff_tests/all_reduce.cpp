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
		auto local_in = in.local();
		buffer_.push_back(*local_in);
		if (buffer_.size() < 2)
			return gff::go_on;
		else {
			int sum = std::accumulate(buffer_.begin(), buffer_.end(), 0);
			REQUIRE(2*NUMBER == sum);
			return gff::eos;
		}
	}

	void svc_init(gff::NDOneToAll &c) {
		c.emit(gam::make_public<int>(NUMBER));
	}

	void svc_end(gff::NDOneToAll &c) {
	}
private:
	vector< int > buffer_;
	unsigned int iter_ = 2;
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

	gff::NDOneToAll all;

	for (unsigned i = 0; i < NWORKERS; i++)
		gff::add(WorkerAllReduce(all,all));

	/* execute the network */
	gff::run();
}
