/*
 * mxnet_tensor.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_MXNET_TENSOR_HPP_
#define FAST_FAST_MXNET_TENSOR_HPP_

/* Included only to let eclipse resolve the names */
#include "fast/tensor.hpp"

namespace FAST {

/* Needed to prevent multiple declarations */
template class Tensor<mxnet::cpp::NDArray>;

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor() {
	bt = mxnet::cpp::NDArray();
}

/**
 * Constructor from raw data and shape
 * @param raw_data
 * @param shape
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(float * raw_data, vector<unsigned int> shape) {
	bt = mxnet::cpp::NDArray(raw_data, mxnet::cpp::Shape(shape), mxnet::cpp::Context::cpu());
}

/**
 * Constructor from raw data and size only
 * Only for flat data
 * @param raw_data
 * @param shape
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(float * raw_data, size_t size) {
	bt = mxnet::cpp::NDArray(raw_data, mxnet::cpp::Shape(size), mxnet::cpp::Context::cpu());
}

/**
 * Copy constructor from Tensor with same template argument
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(Tensor<mxnet::cpp::NDArray> & t) {
	bt = t.getFrameworkObject();
}

/**
 *
 * @return a vector with each dimension of the tensor
 */
template <>
template <typename... Args>
float Tensor<mxnet::cpp::NDArray>::at(Args... args) const {
	return bt.At(args...);
}

/**
 *
 * @return raw pointer of aligned tensor data for NDArray
 */
template <>
const float * Tensor<mxnet::cpp::NDArray>::getRawPtr() {
	return bt.GetData();
}

/**
 *
 * @return a vector with each dimension of the tensor
 */
template <>
vector<unsigned int> Tensor<mxnet::cpp::NDArray>::getShape() const {
	return bt.GetShape();
}

/**
 *
 * @return a vector with each dimension of the tensor
 */
template <>
void Tensor<mxnet::cpp::NDArray>::setShape(vector<unsigned int> shape) {
	bt.Reshape(mxnet::cpp::Shape(shape));
}

/**
 *
 * @return a vector with flattened tensor values
 */
template <>
vector<float> Tensor<mxnet::cpp::NDArray>::getStdValues() {
	vector<float> out;
	bt.SyncCopyToCPU(&out);
	return out;
}

}


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */
