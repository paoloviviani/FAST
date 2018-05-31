/*
 * tensor_wrapper.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_TENSOR_WRAPPER_HPP_
#define INCLUDE_FAST_TENSOR_WRAPPER_HPP_

#include <vector>
using namespace std;

namespace FAST {

/**
 * Tensor interface to specialized sub-classes
 * Using Curiously recurring template pattern
 * https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * Equivalent to abstract class with virtual methods, but with compile time resolution
 */
template <typename Tderived>
class TensorWrapper {
public:
	vector<int> getShape() const { return static_cast<Tderived*>(this)->getShape();	};
	unsigned long getSize() const { return static_cast<Tderived*>(this)->getSize();	};
protected:
	float * data_handle; //TODO make gam pointer
};

}
#endif /* INCLUDE_FAST_TENSOR_WRAPPER_HPP_ */
