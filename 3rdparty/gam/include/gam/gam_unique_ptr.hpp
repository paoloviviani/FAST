/*
 * Copyright (c) 2019 alpha group, CS department, University of Torino.
 *
 * This file is part of gam
 * (see https://github.com/alpha-unito/gam).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @date Apr 14, 2017
 * @brief implements gam_unique_ptr class and some associated functions
 *
 */

#ifndef INCLUDE_GAM_GAM_UNIQUE_PTR_HPP_
#define INCLUDE_GAM_GAM_UNIQUE_PTR_HPP_

#include <memory>

#include "gam/TrackingAllocator.hpp"

namespace gam {

/**
 * @brief represents a unique (local) pointer with custom deleter.
 *
 * A gam_unique_ptr value represents either:
 * - a plain unique pointer with custom deleter
 * - the (local) child of a (global) private pointer
 * In the former case, it is constructed by means of either the new operator or
 * the make_gam_unique function, analogous to std::make_unique.
 * In the latter case, it is constructed by the local() function.
 */
template <typename T>
using gam_unique_ptr = std::unique_ptr<T, void (*)(T *)>;

template <typename _Tp, typename... _Args>
gam_unique_ptr<_Tp> make_gam_unique(_Args &&... __args) {
  return gam_unique_ptr<_Tp>(       //
      NEW<_Tp>(std::forward<_Args>(__args)...),  //
      DELETE<_Tp>);
}

} /* namespace gam */

#endif /* INCLUDE_GAM_GAM_UNIQUE_PTR_HPP_ */
