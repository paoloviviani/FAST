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
 * @brief       3-executor network playing with private-to-public conversion
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
  /* create a public pointer from private, read, push */
  gam::public_ptr<val_t> p(gam::make_private<val_t>(42));
  assert(*p.local() == 42);
  p.push(1);

  /* create a public pointer from from-child private pointer, read, push */
  auto q = gam::make_private<val_t>(42).local();
  gam::public_ptr<val_t> q_(gam::private_ptr<val_t>(std::move(q)));
  assert(*q_.local() == 42);
  q_.push(1);

  /* create and push two private pointers, to be published downstream */
  gam::make_private<val_t>(42).push(1);
  gam::make_private<val_t>(42).push(1);
}

void r1() {
  /* pull twice (from any) */
  auto p = gam::pull_public<val_t>();
  assert(*p.local() == 42);

  p = gam::pull_public<val_t>();
  assert(*p.local() == 42);

  /* pull private pointer (from any), cast it to public pointer and push */
  auto q = gam::pull_private<val_t>();
  p = gam::public_ptr<val_t>(std::move(q));
  p.push(2);

  /* pull private pointer (from any), access, cast to public and push */
  q = gam::pull_private<val_t>();
  auto l = q.local();
  (*l)++;
  p = gam::public_ptr<val_t>(gam::private_ptr<val_t>(std::move(l)));
  p.push(2);
}

void r2() {
  /* pull and forget */
  auto p = gam::pull_public<val_t>(1);  // overwrite
  assert(*p.local() == 42);

  p = gam::pull_public<val_t>(1);  // overwrite
  assert(*p.local() == 43);
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
