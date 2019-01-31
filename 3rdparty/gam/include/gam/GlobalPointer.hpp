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
 *
 * @brief       implements GlobalPointer class
 *
 * @ingroup api
 *
 */
#ifndef INCLUDE_GAM_GLOBALPOINTER_HPP_
#define INCLUDE_GAM_GLOBALPOINTER_HPP_

#include <cstdint>
#include <limits>

#include "gam/Logger.hpp"

namespace gam {

/**
 * GlobalPointer represents a global memory address.
 * Think of the global counterpart of (void *).
 *
 * Internally it is composed by a 64-bit descriptor.
 * Value 0 for the descriptor is reserved for the runtime.
 * Values for the descriptor within the range:
 * [first_reserved, std::numeric_limits<uint64_t>::max()]
 * are reserved for applications built on top of gam.
 * Reserved descriptor are treated as simple numeric values rather than (global)
 * memory addresses. For instance, no releasing mechanism is triggered upon
 * destruction of the enclosing GlobalPointer.
 *
 * 64-bit descriptor is composed by:
 * - 1-bit  reserved (0 = address, 1 = reserved)
 * - 31-bit home partition
 * - 32-bit offset
 */
class GlobalPointer {
 public:
  static constexpr uint64_t first_reserved = (uint64_t)1 << 63;
  static constexpr uint64_t last_reserved =  //
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t max_home = ((uint64_t)1 << 31) - 1;

  GlobalPointer() {}

  GlobalPointer(uint64_t lsb, executor_id home) {
    assert(home <= max_home);  // todo error reporting
    descriptor_ = lsb | ((uint64_t)home << 32);

    /* check consistency */
    assert(is_address());
    assert((lsb | ((uint64_t)home << 32)) == this->address());
    assert(home == this->home());
  }

  GlobalPointer(uint64_t d) { descriptor_ = d; }

  bool operator==(const GlobalPointer& gp) {
    return this->address() == gp.address();
  }

  /*
   * check if the pointer represents a global address
   *
   * @retval TRUE if global address
   * @retval FALSE if reserved value
   */
  bool is_address() const {
    return (descriptor_ != 0) && (descriptor_ < first_reserved);
  }

  /*
   ***************************************************************************
   *
   * getters block
   *
   ***************************************************************************
   */
  inline uint64_t address() const { return descriptor_; }

  /*
   ***************************************************************************
   *
   * setters block
   *
   ***************************************************************************
   */
  inline void address(uint64_t d) { descriptor_ = d; }

  /**
   * @brief pretty-prints the pointer
   */
  friend std::ostream& operator<<(std::ostream& out, const GlobalPointer& f) {
    if (f.is_address())
      out << "{addr=" << f.address() << " home=" << f.home() << "}";
    else
      out << "{token=" << f.address() << "}";
    return out;
  }

 private:
  uint64_t descriptor_ = 0;

  inline executor_id home() const { return (executor_id)(descriptor_ >> 32); }
};

} /* namespace gam */

#endif /* INCLUDE_GAM_GLOBALPOINTER_HPP_ */
