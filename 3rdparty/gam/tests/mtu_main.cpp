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
 * @brief       testing compilation with multiple translation units - main
 *
 */

#include <cassert>
#include <iostream>

#include "gam.hpp"

#include "mtu.h"

typedef int val_t;

void r2() {
  /* pull, do not push */
  auto p = gam::pull_private<val_t>(1);
  assert(p != nullptr);

  /* pull, access, do not push */
  p = gam::pull_private<val_t>(1);  // overwrite
  assert(p != nullptr);
  assert(*p.local() == 44);
}

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char* argv[]) {
  /* rank-specific code */
  switch (gam::rank()) {
    case 0:
      r0();
      break;
    case 1:
      r1();
      break;
    case 2:
      r2();
      break;
  }

  return 0;
}
