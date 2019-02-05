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

	if (argc < 3) {
		std::cout << "Usage: > ./executable grid_height grid_width\n\n";
		return 1;
	}
	size_t grid_h = atoi(argv[1]);
	size_t grid_w = atoi(argv[2]);
	size_t workers = grid_h*grid_w;

	// Row major ordering
	std::vector < std::vector< gff::NondeterminateMerge > > incoming_channels(grid_h);
	std::vector < std::vector< gff::OutBundleBroadcast<gff::NondeterminateMerge> > > outgoing_channels(grid_h);

	for (int i = 0; i < grid_h; i++) {
		for (int j = 0; j < grid_w; j++) {
			incoming_channels.at(i).emplace_back();
		}
	}

	for (unsigned int i = 0; i < grid_h; i++) {
		for (unsigned int j = 0; j < grid_w; j++) {
			outgoing_channels.at(i).emplace_back();
			// Add neighboring channels (i+1,j),(i-1,j),(i,j+1),(i,j-1) in torus topology

			unsigned int up,right,down,left;
			i == grid_h-1 ? down=0 : down=i+1;
			i == 0 ? up=grid_h-1 : up=i-1;
			j == grid_w-1 ? right=0 : right=j+1;
			j == 0 ? left=grid_w-1 : left=j-1;
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(up).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(down).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(right));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(left));
		}
	}

	for (unsigned int i = 0; i < grid_h; i++) {
		for (unsigned int j = 0; j < grid_w; j++) {
			gff::add(MxNetWorker(incoming_channels.at(i).at(j),outgoing_channels.at(i).at(j)));
		}
	}

	/* execute the network */
	gff::run();

	return 0;

}
