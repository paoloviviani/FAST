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
void vecToNDArray(gam_vector<T> * vec, mxnet::cpp::NDArray & t, size_t offset, size_t size, //
					const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu() ) {
	t.SyncCopyFromCPU(vec->data() + offset, size);
}

template <typename T>
void NDVecToVec(std::vector<mxnet::cpp::NDArray> * grad_arrays, const std::vector<std::string> arg_names, gam_vector<T> & vec, //
		std::string inp="X", std::string out="label") {

	size_t grad_size = 0;
	for (size_t i = 0; i < arg_names.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_size += grad_arrays->at(i).Size();
	}
	vec.resize(grad_size);
	for (size_t i = 0; i < arg_names.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		appendToVec(vec, grad_arrays->at(i));
	}

}

template <typename T>
void vecToNDVec(gam_vector<T> & vec, std::vector<mxnet::cpp::NDArray> * grad_arrays, const std::vector<std::string> arg_names, //
		std::string inp="X", std::string out="label", const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {

	size_t offset = 0;
	for (size_t i = 0; i < arg_names.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		vecToNDArray(vec, grad_arrays[i], offset, ctx);
		offset += grad_arrays->at(i).Size();
	}

}

template <typename T>
void accumToNDVec(gam_vector<T> & vec, std::vector<mxnet::cpp::NDArray> * grad_arrays, const std::vector<std::string> arg_names, //
				string inp="X", string out="label", const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {

	size_t offset = 0;
	for (size_t i = 0; i < arg_names.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		grad_arrays->at(i) += mxnet::cpp::NDArray(vec.data() + offset, mxnet::cpp::Shape(grad_arrays->at(i).GetShape()), ctx);
		offset += grad_arrays->at(i).Size();
	}

}

void buildNDVec(std::vector<mxnet::cpp::NDArray> * grad_arrays, const std::vector< std::vector<mx_uint> > grad_shapes, //
		const std::vector<std::string> arg_names, std::string inp="X", std::string out="label", const mxnet::cpp::Context & ctx=mxnet::cpp::Context::cpu()) {
	FAST_DEBUG("DEBUG buildNDVec")
	size_t offset = 0;
	for (size_t i = 0; i < arg_names.size(); i++) {
		if (arg_names[i] == inp || arg_names[i] == out) continue;
		FAST_DEBUG("SIZE: " << grad_arrays->size())
		grad_arrays->push_back( mxnet::cpp::NDArray(mxnet::cpp::Shape(grad_shapes.at(i)), ctx) );
		FAST_DEBUG("SIZE: " << grad_arrays->size())
		grad_arrays->back() = 0;
		FAST_DEBUG("DEBUG 2")
		offset += grad_arrays->back().Size();
		FAST_DEBUG("DEBUG 3")
	}

}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */
