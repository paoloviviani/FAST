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
 * @brief       simple example of 3-executor network exchanging public pointers
 *
 */

#include <cassert>
#include <iostream>

#include "gam.hpp"

typedef int val_t;

/*
 *******************************************************************************
 *
 * rank-specific routines
 *
 *******************************************************************************
 */
void r0() {
  /* create a public pointer (with custom tracking deleter) */
  auto p = gam::make_public<val_t>(42);
  assert(p != nullptr);

  /* access a shared-pointer local copy */
  assert(*p.local() == 42);

  /* push to 1 */
  p.push(1);
}

void r1() {
  /* pull public pointer from 0 */
  auto p = gam::pull_public<val_t>(0);
  assert(p != nullptr);

  /* get a local copy as shared pointer */
  auto lsp = p.local();
  assert(*lsp == 42);

  /* modifies the local copy */
  *lsp = 43;

  /* get a fresh copy of the public pointer */
  gam::public_ptr<val_t> q(p);
  assert(*q.local() == 42);

  /* push both public pointers to executor 2 */
  p.push(2);
  q.push(2);
  q.push(2);  // push twice the same pointer
}

void r2() {
  /* pull public pointer from 1 */
  auto p = gam::pull_public<val_t>(1);
  assert(p != nullptr);
  assert(*p.local());

  /* pull another public pointer from 1 */
  p = gam::pull_public<val_t>(1);  // overwrite
  assert(p != nullptr);

  /* pull again */
  p = gam::pull_public<val_t>(1);  // overwrite
  assert(p != nullptr);
  assert(*p.local() == 42);
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
