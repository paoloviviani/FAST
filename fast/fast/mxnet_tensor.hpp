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
void appendToVec(gam_vector<T> & vec, mxnet::cpp::NDArray & t){
	size_t append_size = t.Size();
	vec.resize(vec.size() + append_size);
	t.SyncCopyToCPU(vec.data() + vec.size(),append_size);
}

template <typename T>
void vecToNDArray(gam_vector<T> & vec, mxnet::cpp::NDArray & t, size_t offset, size_t size, //
					const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu() ) {
	t.SyncCopyFromCPU(vec.data() + offset, size);
}

template <typename T>
void NDVecToVec(std::vector<mxnet::cpp::NDArray> grad_arrays, vector<string> arg_names, gam_vector<T> & vec, //
				string inp="X", string out="label") {

	size_t grad_size = 0;
	for (size_t i = 0; i < arg_names.size(); ++i) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_size += grad_arrays[i].Size();
	}
	vec.resize(grad_size);
	for (size_t i = 0; i < arg_names.size(); ++i) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		appendToVec(vec, grad_arrays[i]);
	}

}

template <typename T>
void VecToNDVec(gam_vector<T> & vec, std::vector<mxnet::cpp::NDArray> grad_arrays, vector<string> arg_names, //
				string inp="X", string out="label", const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {

	size_t offset = 0;
	for (size_t i = 0; i < arg_names.size(); ++i) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_arrays[i] = NDArray(vec.data() + offset, mxnet::cpp::Shape(grad_arrays[i].GetShape()), ctx);
		offset += grad_arrays[i].Size();
	}

}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */
