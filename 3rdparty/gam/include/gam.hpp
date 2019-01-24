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
 * @defgroup api GAM API
 */

/**
 * This file implements the GAM API.
 *
 * @brief       the GAM API
 *
 * @ingroup api
 *
 * @todo non-blocking pull
 * @todo pull from any
 * @todo inlining
 *
 */
#ifndef GAM_HPP_
#define GAM_HPP_

#include "gam/Context.hpp"  //ctx
#include "gam/GlobalPointer.hpp"
#include "gam/TrackingAllocator.hpp"
#include "gam/defs.hpp"
#include "gam/private_ptr.hpp"
#include "gam/public_ptr.hpp"

namespace gam {

/**
 * @brief returns GAM rank
 *
 * The rank of a GAM executor is its index in the space of executors.
 * Namely, it ranges from 1 to the number of executors.
 *
 */
static inline executor_id rank() { return ctx().rank(); }

static inline executor_id cardinality() { return ctx().cardinality(); }

} /* namespace gam */

#endif /* GAM_HPP_ */
