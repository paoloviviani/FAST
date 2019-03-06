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

typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,		//
					gam::public_ptr<FAST::gam_vector<float>>, gam::public_ptr<FAST::gam_vector<float>>, //
					FAST::MXNetWorkerLogic<MODELLOGIC, float>>
	MxNetWorker;

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */

int main(int argc, char **argv)
{
	FAST_LOG_INIT

	if (argc < 2)
	{
		std::cout << "Usage: > ./executable ring size \n\n";
		return 1;
	}
	size_t workers = atoi(argv[1]);
	// size_t workers = 4;

	// Row major ordering
	std::vector<gff::NondeterminateMerge> incoming_channels(workers);
	std::vector<gff::OutBundleBroadcast<gff::NondeterminateMerge>> outgoing_channels(workers);

	for (unsigned int i = 0; i < workers; i++)
	{
		// Add neighboring channels (i+1,j),(i-1,j),(i,j+1),(i,j-1) in torus topology

		unsigned int right, left;
		i == workers - 1 ? right = 0 : right = i + 1;
		i == 0 ? left = workers - 1 : left = i - 1;
		outgoing_channels.at(i).add_comm(incoming_channels.at(right));
		outgoing_channels.at(i).add_comm(incoming_channels.at(left));
	}

	for (unsigned int i = 0; i < workers; i++)
	{
		gff::add(MxNetWorker(incoming_channels.at(i), outgoing_channels.at(i)));
	}

	/* execute the network */
	gff::run();

	return 0;
}
