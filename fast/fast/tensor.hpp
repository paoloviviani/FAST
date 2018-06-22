/*
 * tensor.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_TENSOR_HPP_
#define FAST_FAST_TENSOR_HPP_

#include <vector>
#include <memory>
#include "gam.hpp"
#include "fast/gam_vector.hpp"
#include "mxnet-cpp/MxNetCpp.h"

#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

using namespace std;

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
// Not yet supported
//template <>
//struct is_supported<int32_t> {
//	static const bool value = true;
//};
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

/**
 * Generic tensor class
 * Encapsulates data from a back-end type tensor (e.g. MxNet NDArray)
 * and provides access to raw data and some facilities
 * Meant to be accessible to user
 */
template <typename T = float>
class Tensor {
	/**
	 * Check if template type is supported
	 */
	tensor_type_check(( is_supported<T>::value ));
private:
	/**
	 * Tensor object of deep learning framework
	 */
	gam::private_ptr<gam_vector<T>> data_;
	vector<unsigned int> shape_;

public:
	/*
	 * Dedicated functions prototype. Implement in separate files for different back-ends */
	/**
	 * Constructor from MxNet NDArray, specified in separate file
	 * @param t
	 */
	Tensor(mxnet::cpp::NDArray & t);
	/*
	 * End of dedicated functions */

	/**
	 *
	 * @param t
	 */
	Tensor() {
		data_ = gam::make_private<gam_vector<T>>();
		shape_ = vector<unsigned int>();
	}

	/**
	 * Constructor from raw data and shape
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const T * raw_data, vector<unsigned int> shape) {
		size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned int>());
		data_ = gam::make_private<gam_vector<T>>();
		assert(data_ != nullptr);
		auto data_local = data_.local();
		data_local->resize(size);
		FAST_DEBUG("Created vector with size: " << data_local->size());
		data_local->assign(raw_data, raw_data + size);
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		shape_ = shape;
	}

	/**
	 * Constructor from raw data and size only
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, size_t size) {
		data_ = gam::make_private<gam_vector<T>>();
		assert(data_ != nullptr);
		auto data_local = data_.local();
		data_local->resize(size);
		FAST_DEBUG("Created vector with size: " << data_local->size());
		data_local->assign(raw_data, raw_data + size);
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		shape_.push_back(size);
	}

	Tensor(Tensor<T> & t) {
		size_t size = t.getSize();
		data_ = gam::make_private<gam_vector<T>>();
		assert(data_ != nullptr);
		auto data_local = data_.local();
		FAST_DEBUG("Created vector with shape: " << t.getShape());
		data_local->resize(t.getSize());
		data_local->assign(data_local->data(),data_local->data() + data_local->size());
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
	}

	/**
	 *
	 * @return a vector of unsigned integer with the size of each dimension of the tensor
	 */
	vector<unsigned int> getShape() const { return shape_; };

	/**
	 *
	 */
	void setShape(vector<unsigned int> shape) {  shape_ = shape; };

	/**
	 *
	 * @return the total number of elements of the tensor
	 */
	size_t getSize() const {
		auto v = this->getShape();
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<unsigned int>());
	};

	/**
	 *
	 * @return a vector of unsigned integer with the size of each dimension of the tensor
	 */
	vector<T> getStdValues() {
		auto data_local = data_.local();
		vector<T> out = vector<T>(data_local->begin(),data_local->end());
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}

	/**
	 *
	 * @return private pointer to tensor data
	 */
	gam::private_ptr<gam_vector<T>> getPrivatePtr() {
		return std::move(data_);
	}

	/**
	 * Push private pointer to gam::rank specified
	 * @param to
	 */
	void push(uint32_t to) {
		data_.push(to);
	}

	/**
	 * return offset of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return offset of two dimensions array
	 */
	inline
	size_t Offset(size_t h = 0, size_t w = 0) const {
		return (h *shape_[1]) + w;
	}
	/**
	 * return offset of three dimensions array
	 * \param c channel position
	 * \param h height position
	 * \param w width position
	 * \return offset of three dimensions array
	 */
	inline
	size_t Offset(size_t c, size_t h, size_t w) const {
		return h * shape_[0] * shape_[2] + w * shape_[0] + c;
	}
	/**
	 * return value of the element at (i)
	 * \param i position
	 * \return value of one dimensions array
	 */
	inline
	float at(size_t i) {
		auto data_local = data_.local();
		float out = data_local->at(i);
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}
	/**
	 * return value of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return value of two dimensions array
	 */
	inline
	float at(size_t h, size_t w) {
		auto data_local = data_.local();
		float out = data_local->at(Offset(h, w));
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}
	/**
	 * return value of three dimensions array
	 * \param c channel position
	 * \param h height position
	 * \param w width position
	 * \return value of three dimensions array
	 */
	inline
	float at(size_t c, size_t h, size_t w) {
		auto data_local = data_.local();
		float out = data_local->at(Offset(c, h, w));
		data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}
};

}

#endif /* FAST_FAST_TENSOR_HPP_ */
