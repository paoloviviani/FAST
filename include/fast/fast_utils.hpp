/*
 * fast_utils.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_FAST_UTILS_HPP_
#define FAST_FAST_FAST_UTILS_HPP_

/*
 * Pretty printing for std::vector
 */
template < class T >
inline std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (auto item : v)
    {
        os << item << ", ";
    }
    os << " ]";
    return os;
}

/*
 * Pretty printing for FAST::gam_vector
 */
template < class T >
inline std::ostream& operator << (std::ostream& os, const FAST::gam_vector<T>& v)
{
    os << "[";
    for (auto item : v)
    {
        os << item << ", ";
    }
    os << " ]";
    return os;
}

namespace FAST {


uint32_t rank() {
	return gam::rank();
}

uint32_t cardinality() {
	return gam::cardinality();
}

} /* namespace FAST */

#endif /* FAST_FAST_FAST_UTILS_HPP_ */
