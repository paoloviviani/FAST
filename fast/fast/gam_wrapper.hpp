/*
 * gam_wrapper.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_GAM_WRAPPER_HPP_
#define FAST_FAST_GAM_WRAPPER_HPP_

namespace FAST{

#ifdef USE_GAM

#include "gam.hpp"

static inline uint32_t FAST::rank() {
	return gam::rank();
}

static inline uint32_t FAST::cardinality() {
	return gam::cardinality();
}

#else
static inline uint32_t FAST::rank() {
	return 0;
}

static inline uint32_t FAST::cardinality() {
	return 0;
}
#endif

} // end namespace FAST

#endif /* FAST_FAST_GAM_WRAPPER_HPP_ */
