/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <catch.hpp>
#include <string>
#include <cassert>
#include <cmath>
#include <stdlib.h>

#include <gff.hpp>
#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

#define CATCH_CONFIG_MAIN

Context ctx = Context::cpu();  // Use CPU for training

struct Dummy {
	std::vector<NDArray> grad_arrays;
};

class ModelLogic {
public:
	void init() {
		Context ctx = Context::cpu();  // Use CPU for training

		exec = new Dummy();
		arg_names.push_back("first");
//		arg_names.push_back("second");
//		arg_names.push_back("third");
//		arg_names.push_back("fourth");

		for (size_t i = 0; i < arg_names.size(); ++i) {
			exec->grad_arrays.push_back( NDArray(Shape(8, 2), ctx) );
			exec->grad_arrays[i] = 0.;
			FAST_DEBUG("(LOGIC INIT): gradients initial values = " << exec->grad_arrays[i])
		}
		FAST_INFO("Logic initialized")
	}


	void run_batch() {
		FAST_INFO("(LOGIC): run batch, iteration = " << iter_)
		for (size_t i = 0; i < arg_names.size(); ++i) {
			exec->grad_arrays[i] += 0.1;
			FAST_DEBUG("(LOGIC RUN): gradients new values = " << exec->grad_arrays[i])
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		iter_++;
		if (iter_ == 150)
			max_epoch_reached = true; // Terminate
	}

	void update(std::vector<mxnet::cpp::NDArray> &in) {
		REQUIRE(in.size() > 0);
		for (size_t i = 0; i < arg_names.size(); ++i) {
			FAST_DEBUG("(LOGIC UPDATE): original gradients = " << exec->grad_arrays[i])
			FAST_DEBUG("(LOGIC UPDATE): incoming gradients = " << in[i])
			exec->grad_arrays[i] += in[i];
			FAST_DEBUG("(LOGIC UPDATE): updated gradients values = " << exec->grad_arrays[i])
		}
		FAST_INFO("(LOGIC UPDATE): updated")
	}

	void finalize() {

	}

	Dummy * exec;
	vector<string> arg_names;
	unsigned int iter_ = 0;
	bool max_epoch_reached = false;
};

typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, gam::public_ptr< FAST::gam_vector<float> >, //
		FAST::MXNetWorkerLogic<ModelLogic, float> > MxNetWorker;

/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE( "MxNet worker basic test", "gam,gff,multi,mxnet" ) {
	FAST_LOG_INIT
	FAST_INFO("TEST name: "<< Catch::getResultCapture().getCurrentTestName());

	gff::NondeterminateMerge to_one, to_two;
	gff::OutBundleBroadcast<gff::NondeterminateMerge> one, two;

	one.add_comm(to_two);
	two.add_comm(to_one);

	gff::add(MxNetWorker(to_one,one));
	gff::add(MxNetWorker(to_two,two));

	/* execute the network */
	gff::run();

}
