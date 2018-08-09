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

#define NWORKERS    4
#define NUMBER      25

/*
 * To define a gff node, the user has to define first an internal class (a.k.a.
 * dff logic), with the following functions:
 * - svc_init: called once at the beginning of node's execution
 * - svc_end:  called once at the end of node's execution
 * - svc:      called repeatedly during node's execution
 *
 * The signature of svc varies depending on node's family (source, processor,
 * sink). See sample codes.
 *
 * To complete the definition of a gff node, the internal class is passed as
 * template parameter to the generic class corresponding to the node's family.
 */

/*
 *******************************************************************************
 *
 * farm components
 *
 *******************************************************************************
 */
/*
 * gff logic generating a stream of random integers
 */
class EmitterLogic {
public:
	EmitterLogic() {
	}

	/**
	 * The svc function is called repeatedly by the runtime, until an eos
	 * token is returned.
	 * Pointers are sent downstream by calling the emit() function on the
	 * output channel, that is passed as input argument.
	 *
	 * @param c is the output channel (could be a template for simplicity)
	 * @return a gff token
	 */
	gff::token_t svc(gff::OneToAll &c) {
		c.emit(gam::make_public<int>(NUMBER));
		return gff::eos;
	}

	void svc_init() {
	}

	void svc_end(gff::OneToAll &c) {
		REQUIRE(true);
	}

private:
};

/*
 * define a Source node with the following template parameters:
 * - the type of the output channel
 * - the type of the emitted pointers
 * - the gff logic
 */
typedef gff::Source<gff::OneToAll, //
		gam::public_ptr<int>, //
		EmitterLogic> Emitter;

/*
 * gff logic low-passing input integers below THRESHOLD and computes sqrt
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
	gff::token_t svc(gam::public_ptr<int> &in, gff::NondeterminateMerge &c) {
		auto local_in = in.local();
		REQUIRE(NUMBER == *local_in);
		c.emit(gam::make_private<char>((char) std::sqrt(*local_in)));
		return gff::go_on;
	}

	void svc_init(gff::NondeterminateMerge &c) {
	}

	void svc_end(gff::NondeterminateMerge &c) {
	}
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the output channel
 * - the type of the input pointers
 * - the type of the output pointers
 * - the gff logic
 */
typedef gff::Filter<gff::OneToAll, gff::NondeterminateMerge, //
		gam::public_ptr<int>, gam::private_ptr<char>, //
		WorkerLogic> Worker;

/*
 * gff logic summing up all filtered tokens and finally checking the result
 */
class CollectorLogic {
public:
	CollectorLogic() {}

	/**
	 * The svc function is called upon each incoming pointer from upstream.
	 *
	 * @param in is the input pointer
	 * @return a gff token
	 */
	void svc(gam::private_ptr<char> &in) {
		auto local_in = in.local();
		REQUIRE(std::sqrt(NUMBER) == *local_in);
		sum += *local_in;
	}

	void svc_init() {
	}

	/*
	 * at the end of processing, check the result
	 */
	void svc_end() {
		int res = std::sqrt(NUMBER)*NWORKERS;
 		REQUIRE(res == sum);
	}

private:
	int sum = 0;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the input pointers
 * - the gff logic
 */
typedef gff::Sink<gff::NondeterminateMerge, //
		gam::private_ptr<char>, //
		CollectorLogic> Collector;
/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */
TEST_CASE( "gff basic broadcast", "gam,gff" ) {
	/*
	 * Create the channels for inter-node communication.
	 * A channel can carry both public and private pointers.
	 */
	gff::OneToAll e2w;
	gff::NondeterminateMerge w2c;

	/*
	 * In this preliminary implementation, a single global network is
	 * created and nodes can be added only to the global network.
	 */

	/* bind nodes to channels and add to the network */
	gff::add(Emitter(e2w)); //e2w is the emitter's output channel
	for (unsigned i = 0; i < NWORKERS; ++i)
		gff::add(Worker(e2w, w2c)); //e2w/w2c are the workers' i/o channels
	gff::add(Collector(w2c)); //w2c is the collector's input channel

	/* execute the network */
	gff::run();
}
