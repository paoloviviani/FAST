/*
 * 2D_torus.cpp
 *
 *  Created on: Jan 29, 2019
 *      Author: pvi
 */

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <gff.hpp>
#include <fast.hpp>

#ifndef MODELLOGIC
#define MODELLOGIC ModelLogic
#endif

typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,//
		gam::public_ptr< FAST::gam_vector<float> >, gam::public_ptr< FAST::gam_vector<float> >, //
		FAST::MXNetWorkerLogic<MODELLOGIC, float> > MxNetWorker;

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */

int main(int argc, char** argv) {
	FAST_LOG_INIT

	gff::NondeterminateMerge to_one, to_two;

	gff::add(MxNetWorker(to_one,to_two));
	gff::add(MxNetWorker(to_two,to_one));

	/* execute the network */
	gff::run();

	return 0;


}
