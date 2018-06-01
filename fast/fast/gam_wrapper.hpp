/*
 * gam_wrapper.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_GAM_WRAPPER_HPP_
#define FAST_FAST_GAM_WRAPPER_HPP_

namespace FAST{
uint32_t rank();
}

#ifdef USE_GAM

#include "gam.hpp"

inline uint32_t FAST::rank() {
	return gam::rank();
}

#else
uint32_t FAST::rank() {
	return 0;
}
#endif

#endif /* FAST_FAST_GAM_WRAPPER_HPP_ */
