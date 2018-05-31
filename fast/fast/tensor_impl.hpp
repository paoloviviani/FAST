/*
 * tensor_impl.hpp
 *
 *  Created on: May 31, 2018
 *      Author: pvi
 */

#ifndef FAST_TENSOR_IMPL_HPP_
#define FAST_TENSOR_IMPL_HPP_

#include "fast/tensor.hpp"

#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

namespace FAST{



/**
 *
 * @param tensor
 */
template <typename backendTensor>
Tensor<backendTensor>::Tensor(backendTensor tensor){
	tensor_type_check(( is_supported<backendTensor>::value ));
	backend_tensor = tensor;
}

/**
 * Only enters if tensor type is supported
 * @return vector of unsigned integers with the shape of the tensor
 */
template <typename backendTensor>
vector<unsigned int> Tensor<backendTensor>::getShape() const {
	tensor_type_check(( is_supported<backendTensor>::value  ));

	if ( is_MxNDAarray<backendTensor>::value ){
		return backend_tensor.GetShape();
	}
}

}

#endif /* FAST_TENSOR_IMPL_HPP_ */
