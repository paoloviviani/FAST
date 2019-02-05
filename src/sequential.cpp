/*
 * 2D_torus.cpp
 *
 *  Created on: Jan 29, 2019
 *      Author: pvi
 */

#include <iostream>
#include <string>

#ifndef MODELLOGIC
#define MODELLOGIC ModelLogic
#endif

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */

int main(int argc, char** argv) {

	ModelLogic logic;

	logic.init();

	while (!logic.max_epoch_reached) {
		logic.run_batch();
	}

	logic.finalize();

	return 0;


}
