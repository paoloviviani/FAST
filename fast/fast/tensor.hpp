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
class Tensor : public TensorWrapper {
	/**
	 * Check if template type is supported
	 */
	tensor_type_check(( is_supported<backendType>::value ));
protected:
	/**
	 * Tensor object of deep learning framework
	 */
	std::unique_ptr<float> data_;
	vector<unsigned int> shape_;

public:

	/**
	 *
	 * @param t
	 */
	Tensor() {
		data_ = std::unique_ptr<float>();
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
		data_ = std::unique_ptr<float>(new float[size]);
		std::copy(raw_data, raw_data + size, data_.get());
		shape_ = shape;
	}

	/**
	 * Constructor from raw data and size only
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, size_t size) {
		data_ = std::unique_ptr<float>(new float[size]);
		std::copy(raw_data, raw_data + size, data_.get());
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
		vector<float> out(data_.get(),data_.get()+this->getSize());
		return out;
	}

	/**
	 *
	 * @return raw pointer to tensor data
	 */
	const float * getRawPtr() {return data_.get();};

	/**
	 *
	 * @return the element at the given index, based on underlying implementation
	 */
	template<typename... Args>
	float at(Args...) const;

	/**
	 *
	 * @return the object of type defined by the deep learning framewrok
	 */
	backendType getFrameworkObject();

};

}

#endif /* FAST_FAST_TENSOR_HPP_ */
