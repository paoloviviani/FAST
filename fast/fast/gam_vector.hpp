/*
 * gam_array.hpp
 *
 *  Created on: Jun 20, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_GAM_VECTOR_HPP_
#define FAST_FAST_GAM_VECTOR_HPP_

namespace FAST {

template<typename T>
struct gam_vector : public std::vector<T> {
	using vsize_t = typename std::vector<T>::size_type;
	vsize_t size_ = 0;

	gam_vector() = default;

	/* ingesting constructor */
	template<typename StreamInF>
	gam_vector(StreamInF &&f) {
		typename std::vector<T>::size_type in_size;
		f(&in_size, sizeof(vsize_t));
		this->resize(in_size);
		assert(this->size() == in_size);
		f(this->data(), in_size * sizeof(T));
	}

	/* marshalling function */
	gam::marshalled_t marshall() {
		gam::marshalled_t res;
		size_ = this->size();
		res.emplace_back(&size_, sizeof(vsize_t));
		res.emplace_back(this->data(), size_ * sizeof(T));
		return res;
	}
};

} //Namespace FAST

#endif /* FAST_FAST_GAM_VECTOR_HPP_ */
