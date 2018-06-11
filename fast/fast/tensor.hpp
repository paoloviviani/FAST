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
#include "mxnet-cpp/MxNetCpp.h"
#include "gam.hpp"

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
struct is_supported<mxnet::cpp::NDArray> {
	static const bool value = true;
};


namespace FAST {

/**
 * Generic tensor class
 * Encapsulates a back-end type tensor (e.g. MxNet NDArray)
 * and provides access to raw data and some facilities
 * Meant to be accessible to user
 */
template <typename backendType>
class Tensor {
	/**
	 * Check if template type is supported
	 */
	tensor_type_check(( is_supported<backendType>::value ));
protected:
	/**
	 * Tensor object of deep learning framework
	 */
	gam::private_ptr<vector<float>> data_;
	vector<unsigned int> shape_;

public:

	/**
	 *
	 * @param t
	 */
	Tensor() {
		data_ = gam::private_ptr<vector<float>>();
		shape_ = vector<unsigned int>();
	}

	/**
	 * Constructor from generic framework tensor
	 * @param t
	 */
	Tensor(backendType t);

	/**
	 * Constructor from raw data and shape
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, vector<unsigned int> shape) {
		size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned int>());
		data_ = gam::make_private<vector<float>>(size);
		FAST_DEBUG("Created vector with size: " << data_.local()->size());
//		data_.local()->assign(raw_data, raw_data + size);
		std::copy(raw_data, raw_data + size, data_.local()->data());
		shape_ = shape;
	}

	/**
	 * Constructor from raw data and size only
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, size_t size) {
		data_ = gam::make_private<vector<float>>(size);
		FAST_DEBUG("Created vector with size: " << data_.local()->size());
//		data_.local()->assign(raw_data, raw_data + size);
		std::copy(raw_data, raw_data + size, data_.local()->data());
		shape_.push_back(size);
	}

	/**
	 * Copy constructor from NxNet NDArray
	 * @param t
	 */
	Tensor(Tensor<mxnet::cpp::NDArray> & t);

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
	vector<float> getStdValues() {
		return vector<float>(*data_.local());
	}

	/**
	 *
	 * @return raw pointer to tensor data
	 */
	const float * getRawPtr() {
		return data_.local()->data();
	}

	/**
	 *
	 * @return private pointer to tensor data
	 */
	gam::private_ptr<vector<float>> getPrivatePtr() {
		return std::move(data_);
	}

	/**
	 *
	 * @return the element at the given index, based on underlying implementation
	 */
	template<typename... Args>
	float at(Args...);

	/**
	 *
	 * @return the object of type defined by the deep learning framework
	 */
	backendType getFrameworkObject();

};

}

#endif /* FAST_FAST_TENSOR_HPP_ */
