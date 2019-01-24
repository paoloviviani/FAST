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
 * @date Mar 19, 2017
 * @brief defines global types
 */

#ifndef INCLUDE_GAM_DEFS_HPP_
#define INCLUDE_GAM_DEFS_HPP_

#include <cstdint>
#include <memory>
#include <vector>

namespace gam {

/**
 * @brief access level for global pointer
 *
 * AccessLevel defines access rights for global pointers by an executor:
 * - AL_NIL not accessible by the executor
 * - AL_PUBLIC read-only accessible by all executors
 * - AL_PRIVATE only accessible by the owner
 */
enum AccessLevel { AL_PUBLIC, AL_PRIVATE };

///**
// * @brief role in a executor-to-executor communication
// *
// * Role defines the role of an executor within an executor-to-executor
// communication:
// * - CR_PRODUCER executor acts as producer
// * - CR_CONSUMER executor acts as consumer
// */
// enum Role {
//    CR_PRODUCER, CR_CONSUMER
//};

typedef uint32_t executor_id;

template <typename T>
static void nop_deleter(T *) {}

template <typename T>
void default_deleter(T *p) {
  delete p;
}

struct marshalled_entry {
  void *base;
  size_t size;
  marshalled_entry(void *base_, size_t size_) : base(base_), size(size_) {}
};
using marshalled_t = std::vector<marshalled_entry>;

} /* namespace gam */

#endif /* INCLUDE_GAM_DEFS_HPP_ */
