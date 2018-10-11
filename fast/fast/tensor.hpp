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
#include <future>
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
	gam::public_ptr<gam_vector<T>> data_;
	unsigned long long size_;

public:

	/**
	 *
	 * @param t
	 */
	Tensor() {
		data_ = gam::make_private<gam_vector<T>>();
		size_ = 0;
	}

	/*
	 * Dedicated functions prototype. Implement in separate files for different back-ends
	 */
#ifdef MXNET_TENSOR
	/**
	 * Constructor from MxNet NDArray, specified in separate file
	 * @param t
	 */
	Tensor(mxnet::cpp::NDArray & t);
	/**
	 * Append NDarray to data tensor
	 * @param t
	 */
	void append(mxnet::cpp::NDArray & t);

	mxnet::cpp::NDArray asNDArray(size_t offset, size_t size);
#endif

	/*
	 * End of dedicated functions */

	/**
	 * Constructor from raw data and size
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, size_t size) {
		data_ = gam::make_public<gam_vector<T>>();
		assert(data_ != nullptr);
		auto data_local = data_.local();
		data_local->resize(size);
		data_local->assign(raw_data, raw_data + size);
		data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
		size_ = size;
	}

	Tensor(gam_vector<T> & v) {
		size_ = v.size();
		data_ = gam::make_public<gam_vector<T>>();
		assert(data_ != nullptr);
		auto data_local = data_.local();
		data_local->resize(size_);
		data_local->assign(v.data(),v.data() + size_);
		data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
	}

	/**
	 *
	 * @return the total number of elements of the tensor
	 */
	size_t getSize() const {
		return size_;
	};

	/**
	 *
	 * @return a vector of unsigned integer with the size of each dimension of the tensor
	 */
	vector<T> getStdValues() {
		auto data_local = data_.local();
		vector<T> out = vector<T>(data_local->begin(),data_local->end());
		data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}

	/**
	 *
	 * @return private pointer to tensor data
	 */
	gam::public_ptr<gam_vector<T>> getPrivatePtr() {
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
	 * return value of the element at (i)
	 * \param i position
	 * \return value of one dimensions array
	 */
	inline
	T at(size_t i) {
		auto data_local = data_.local();
		float out = data_local->at(i);
		data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
		return out;
	}
};

template<typename T>
std::unique_ptr<Tensor<T>> pull_tensor(uint32_t from){
	auto p = gam::pull_public<gam_vector<T>>(from);
	auto p_local = p.local();
	unsigned int size = p_local->size();
	return Tensor<T>(p_local->data(),size);
}

template<typename T>
std::unique_ptr<Tensor<T>> pull_tensor(){
	auto p = gam::pull_public<gam_vector<T>>();
	auto p_local = p.local();
	unsigned int size = p_local->size();
	return Tensor<T>(p_local->data(),size);
}

}

#endif /* FAST_FAST_TENSOR_HPP_ */
