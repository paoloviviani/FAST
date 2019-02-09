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

#define NWORKERS    2
#define SIZE      10
#define MAX_ITER      100

/*
 *******************************************************************************
 *
 * farm components
 *
 *******************************************************************************
 */
class WorkerLogic {
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
	gff::token_t svc(gam::public_ptr<FAST::gam_vector<float>> &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

		auto recv = in.local();
		FAST::gam_vector<float> * out = new FAST::gam_vector<float>(SIZE);

		FAST_INFO("Received pointer")

		if (recv->size() == 0) {
			for (int i = 0; i < SIZE; i++) {
				out->at(i) = 1.;
				internal_state_.at(i) = 1.;
			}
			auto out_ptr = gam::public_ptr< FAST::gam_vector<float> >(out, [](FAST::gam_vector<float> * ptr){delete ptr;});
			c.emit(out_ptr);
			iter++;
			return gff::go_on;
		}

		for (int i = 0; i < SIZE; i++) {
			internal_state_.at(i) += recv->at(i);
			out->at(i) = 1.;
		}

		iter++;
		if (iter == MAX_ITER && FAST::rank() == 0)
			return gff::eos;
		if (iter == MAX_ITER && FAST::rank() != 0)
					return gff::go_on;

		auto out_ptr = gam::public_ptr< FAST::gam_vector<float> >(out, [](FAST::gam_vector<float> * ptr){delete ptr;});
		c.emit(out_ptr);

		return gff::go_on;
	}

	void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		internal_state_.resize(10);

		FAST::gam_vector<float> * ptr = new FAST::gam_vector<float>(0);
		auto dummy_out = gam::public_ptr< FAST::gam_vector<float> >(ptr, [](FAST::gam_vector<float> * ptr){delete ptr;});
		FAST_INFO("Emitting trigger")

		c.emit(dummy_out);
	}

	void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
		float test = MAX_ITER;
		REQUIRE(internal_state_.at(0) == test);
		FAST_INFO("(svc_end): intenral_state = " << internal_state_)
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	}
private:
	vector< float > internal_state_;
	int iter = 0;
};


typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, gam::public_ptr< FAST::gam_vector<float> >, //
		WorkerLogic > TrainingWorker;
/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "gff training mockup", "gam,gff,multi" ) {
	FAST_LOG_INIT
	FAST_INFO(Catch::getResultCapture().getCurrentTestName());

	gff::NondeterminateMerge to_one, to_two;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two;

	one.add_comm(to_two);
	two.add_comm(to_one);

	gff::add(TrainingWorker(to_one,one));
	gff::add(TrainingWorker(to_two,two));

	/* execute the network */
	gff::run();

}
