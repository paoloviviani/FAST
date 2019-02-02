/*
 * fast_utils.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_FAST_UTILS_HPP_
#define FAST_FAST_FAST_UTILS_HPP_


namespace FAST {


uint32_t rank() {
	return gam::rank();
}

uint32_t cardinality() {
	return gam::cardinality();
}

} /* namespace FAST */

#endif /* FAST_FAST_FAST_UTILS_HPP_ */
