/*
 * example_unit_test.cpp
 *
 *  Created on: Mar 31, 2018
 *      Author: viviani
 */

/*
 * dedicated one file to compile the source code of Catch itself and
 * reuse the resulting object file for linking.
 *
 *useful to minimize the compile time
 */

// Let Catch provide main():

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

