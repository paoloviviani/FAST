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
void insert(gam_vector<T> & in, mxnet::cpp::NDArray & t){
	size_t append_size = t.Size();
	t.SyncCopyToCPU(in.data() + in.size(),append_size);
}

template <typename T>
void append(gam_vector<T> & in, mxnet::cpp::NDArray & t){
	size_t append_size = t.Size();
	in.resize(in.size() + append_size);
	t.SyncCopyToCPU(in.data() + in.size(),append_size);
}

template <typename T>
void extract(gam_vector<T> & in, mxnet::cpp::NDArray & t, size_t pos, size_t size, const mxnet::cpp::Context & ctx) {
	t.SyncCopyFromCPU(in.data() + pos, size);
}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */
