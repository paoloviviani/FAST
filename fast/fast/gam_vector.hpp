/*
 * gam_array.hpp
 *
 *  Created on: Jun 20, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_GAM_VECTOR_HPP_
#define FAST_FAST_GAM_VECTOR_HPP_


#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

/**
 * Statically check if type is supported
 */
template <typename T>
struct is_supported {
	static const bool value = false;
};

template <>
struct is_supported<float> {
	static const bool value = true;
};

template <>
struct is_supported<int32_t> {
	static const bool value = true;
};
//
//template <>
//struct is_supported<int8_t> {
//	static const bool value = true;
//};
//
//template <>
//struct is_supported<bool> {
//	static const bool value = true;
//};

namespace FAST {

template<typename T>
struct gam_vector : public vector<T> {

	tensor_type_check(( is_supported<T>::value ));

	using vsize_t = typename std::vector<T>::size_type;
	vsize_t size_ = 0;

	explicit gam_vector() = default;
	using vector<T>::vector;

	gam_vector(const std::vector<T>& in) : vector<T>(in) {}

	/* ingesting constructor */
	template<typename StreamInF>
	void ingest(StreamInF &&f) {
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
