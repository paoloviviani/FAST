/*
 * gam_wrapper.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_GAM_WRAPPER_HPP_
#define FAST_FAST_GAM_WRAPPER_HPP_

#include "gam/include/gam.hpp"

namespace FAST{

uint32_t rank() {
	return gam::rank();
}

}

#endif /* FAST_FAST_GAM_WRAPPER_HPP_ */
