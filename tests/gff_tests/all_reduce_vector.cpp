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
	gff::token_t svc(gam::public_ptr< FAST::gam_vector<float> > &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		auto local_in = in.local();
		auto vec = *local_in;
		in.reset();
		iter++;
		FAST_DEBUG("Received vector " << vec << "  " << iter << " times.");
		buffer_.push_back(vec);
		if (buffer_.size() == 2) {
			for (int i = 0; i < buffer_[0].size(); i++)
				sum_.at(i) = buffer_.at(0).at(i) + buffer_.at(1).at(i);
			while (out.use_count() > 1) {
				sleep(0.01);
			}
			return gff::eos;
		}
		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		FAST::gam_vector<float> local_out;
		local_out.push_back(NUMBER);
		local_out.push_back(NUMBER);
		out = gam::make_public<FAST::gam_vector<float>>(local_out);
		c.emit(out);
		FAST_DEBUG("Emitted vector " << local_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		REQUIRE(2*NUMBER == sum_.at(0));
	}
private:
	vector< FAST::gam_vector<float> > buffer_;
	int iter = 0;
	FAST::gam_vector<float> sum_ = {0.,0.};
	gam::public_ptr< FAST::gam_vector<float> > out = nullptr;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the output channel
 * - the type of the input pointers
 * - the type of the output pointers
 * - the gff logic
 */
typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, 	gam::public_ptr< FAST::gam_vector<float> >, //
		WorkerLogicAllReduce> WorkerAllReduce;
/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "gff allreduce multi vector", "gam,gff,multi" ) {
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());

	gff::NondeterminateMerge to_one, to_two, to_three;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two, three;

	one.add_comm(to_two);
	one.add_comm(to_three);
	two.add_comm(to_one);
	two.add_comm(to_three);
	three.add_comm(to_two);
	three.add_comm(to_one);

	gff::add(WorkerAllReduce(to_one,one));
	gff::add(WorkerAllReduce(to_two,two));
	gff::add(WorkerAllReduce(to_three,three));

	/* execute the network */
	gff::run();

}
