/*
 * mxnet_tensor.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_MXNET_TENSOR_HPP_
#define FAST_FAST_MXNET_TENSOR_HPP_

/* Included only to let eclipse resolve the names */
#include "mxnet-cpp/MxNetCpp.h"
#include "fast/gam_vector.hpp"

namespace FAST {

template <typename T>
void appendToVec(gam_vector<T> & vec, mxnet::cpp::NDArray & t, const float factor=1.){
	size_t append_size = t.Size();
	size_t original_size = vec.size();
	vec.resize(vec.size() + append_size);
	(t*factor).SyncCopyToCPU(vec.data() + original_size,append_size);
}

template <typename T>
void vecToNDArray(gam_vector<T> & vec, mxnet::cpp::NDArray & t, size_t offset, size_t size, //
					const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu() ) {
	t.SyncCopyFromCPU(vec.data() + offset, size);
}

template <typename T>
void NDVecToVec(std::vector<mxnet::cpp::NDArray> & grad_arrays, const std::vector<std::string> arg_names, gam_vector<T> & vec, //
		std::string inp="X", std::string out="label", const float factor=1.) {

	size_t grad_size = 0;
	for (size_t i = 0; i < grad_arrays.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_size += grad_arrays[i].Size();
	}
	vec->reserve(grad_size);
	for (size_t i = 0; i < grad_arrays.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out)	continue;
		appendToVec(vec, grad_arrays[i], factor);
	}

}

template <typename T>
void vecToNDVec(gam_vector<T> & vec, std::vector<mxnet::cpp::NDArray> & grad_arrays, const std::vector<std::string> arg_names, //
		std::string inp="X", std::string out="label",//
		const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {

	size_t offset = 0;
	for (size_t i = 0; i < grad_arrays.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out)	continue;
		vecToNDArray(vec, grad_arrays[i], offset, grad_arrays[i].Size(), ctx);
		offset += grad_arrays[i].Size();
	}

}

template <typename T>
void accumToNDVec(gam_vector<T> & vec, std::vector<mxnet::cpp::NDArray> & grad_arrays, const std::vector<std::string> arg_names,//
		 const std::string inp="X", const std::string out="label", const float decay=1., //
		 const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {
	size_t offset = 0;
	for (size_t i = 0; i < grad_arrays.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_arrays[i] += mxnet::cpp::NDArray(vec.data() + offset, mxnet::cpp::Shape(grad_arrays[i].GetShape()), ctx);
		offset += grad_arrays[i].Size();
	}

}

void buildNDVec(std::vector<mxnet::cpp::NDArray> & grad_arrays, const std::vector<mxnet::cpp::NDArray> & exec_grads, //
		const std::vector<std::string> arg_names, const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {

	for (size_t i = 0; i < arg_names.size(); i++) {
		grad_arrays.push_back( mxnet::cpp::NDArray(mxnet::cpp::Shape(exec_grads[i].GetShape()), ctx) );
		grad_arrays.back() = 0;
	}

}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */
