/* ***************************************************************************
 *
 *  This file is part of gam.
 *
 *  gam is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with gam. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/*
 * network.hpp
 *
 *  Created on: Jul 26, 2018
 *      Author: Maurizio Drocco, modified by Paolo Viviani
 */

#ifndef FAST_FAST_NETWORK_HPP_
#define FAST_FAST_NETWORK_HPP_


#include <cassert>
#include "gam.hpp"

namespace FAST {

/*
 * Singleton class representing a whole GFF application.
 */
class Network {
public:
	static Network *getNetwork() {
		static Network network;
		return &network;
	}

	~Network() {
		std::vector<Worker *>::size_type i;
		for (i = 0; i < gam::rank() && i < cardinality(); ++i)
			delete nodes[i];
		if (i < cardinality())
			assert(nodes[i] == nullptr);
		for (++i; i < cardinality(); ++i)
			delete nodes[i];
		nodes.clear();
	}

	gam::executor_id cardinality() {
		return nodes.size();
	}

	template<typename T>
	void add(const T &n) {
		auto np = dynamic_cast<Worker *>(new T(n));
		np->id((gam::executor_id) nodes.size());
		nodes.push_back(np);
	}

	template<typename T>
	void add(T &&n) {
		auto np = dynamic_cast<Worker *>(new T(std::move(n)));
		np->id((gam::executor_id) nodes.size());
		nodes.push_back(np);
	}

	void run() {
		/* initialize the logger */
		char *env = std::getenv("GAM_LOG_PREFIX");
		assert(env);
		FAST_LOG_INIT;

		/* check cardinality */
		if (gam::cardinality() < cardinality()) {
			FAST_ERROR("Executor id > network cardinality")
			assert(false);
		}

		if (gam::rank() < cardinality()) {
			/* run the node associated to the executor */
			nodes[gam::rank()]->run();

			/* call node destructor to trigger destruction of data members */
			delete nodes[gam::rank()];
			nodes[gam::rank()] = nullptr;

		}

		/* finalize the logger */
		FAST_LOG_FINALIZE;
	}

private:
	std::vector<Worker *> nodes;
};

template<typename T>
static void add(const T &n) {
	Network::getNetwork()->add(n);
}

template<typename T>
static void add(T &&n) {
	Network::getNetwork()->add(std::move(n));
}

static void run() {
	Network::getNetwork()->run();
}

}// namespace FAST

#endif /* FAST_FAST_NETWORK_HPP_ */
