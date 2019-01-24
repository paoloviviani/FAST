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
 * @brief       implements private_ptr and some related functions
 *
 * @ingroup api
 *
 */
#ifndef INCLUDE_GAM_PRIVATE_PTR_HPP_
#define INCLUDE_GAM_PRIVATE_PTR_HPP_

#include "gam/Context.hpp"  //ctx
#include "gam/GlobalPointer.hpp"
#include "gam/Logger.hpp"
#include "gam/gam_unique_ptr.hpp"

namespace gam {

/**
 * @brief class representing private pointers.
 */
template <typename T>
class private_ptr {
 public:
  typedef T element_type;

  private_ptr() noexcept {}

  /**
   * @brief private pointer constructor from local pointer with custom deleter.
   *
   * It constructs a private pointer by wrapping a local pointer.
   * The deleter is used to free the local pointer.
   *
   * @param lp the local pointer to wrap
   * @param d the deleter
   */
  template <typename Deleter>
  private_ptr(T *const lp, Deleter d) {
    if (lp) {
      LOGLN_OS("PVT constructor local=" << lp);
      if (!make(lp, d))
        std::cerr << "> could not create a private pointer for local pointer: "
                  << lp << std::endl;
    }
  }

  /**
   * @brief disruptive private pointer constructor from local unique pointer.
   *
   * The unique constructor builds a brand new private pointer from a unique
   * pointer.
   * In case the unique pointer is child of a private pointer (see \ref
   * gam_unique_ptr), it is destroyed and its memory is recycled for the new
   * pointer.
   *
   * This constructor represents moving local to global memory.
   *
   * @param lup the unique local pointer
   */
  private_ptr(gam_unique_ptr<T> &&lup) {
    T *lp = lup.get();

    if (lup != nullptr) {
      LOGLN_OS("PVT constructor unique=" << lp);

      /* lookup parent global pointer */
      if (!ctx().has_parent(lp)) {
        // not a private child
        if (!make(lp, lup.get_deleter())) {
          std::cerr
              << "> could not create a private pointer for gam-unique address: "
              << lp << std::endl;
          return;
        }
      } else {
        // private child: writeback
        if (!writeback(std::move(lup))) {
          std::cerr << "> could not retrieve the private pointer for "
                       "gam-unique address: "
                    << lp << std::endl;
          return;
        }
      }

      /* neutralize destruction */
      lup.release();
    }
  }

  /**
   * @brief private pointer constructor from global pointer.
   * @ingroup internals
   *
   * It constructs a public pointer by wrapping a global pointer.
   *
   * @param p the global pointer to wrap
   */
  private_ptr(const GlobalPointer &p) noexcept : internal_gp(p) {
    if (p.is_address() || p.address() != 0)
      LOGLN_OS("PVT constructor global=" << p);
  }

  ~private_ptr() {
    if (internal_gp.is_address()) {
      LOGLN_OS("PVT destroy global=" << internal_gp);
      reset();
    }
  }

  /*
   ***************************************************************************
   *
   * copy constructor and assignment are deleted
   *
   ***************************************************************************
   */
  private_ptr(const private_ptr &) = delete;
  private_ptr &operator=(const private_ptr &) = delete;

  /*
   ***************************************************************************
   *
   * move constructor and assignment
   *
   ***************************************************************************
   */
  private_ptr(private_ptr &&other) noexcept : internal_gp(other.internal_gp) {
    LOGLN_OS("PVT move-constructor global=" << internal_gp);

    /* neutralize other destruction */
    other.release();
  }

  private_ptr &operator=(private_ptr &&other) noexcept {
    LOGLN_OS("PVT move-assignment obj=" << other << " sub=" << *this);

    /* swap to cause subject destruction */
    GlobalPointer tmp = internal_gp;
    internal_gp = other.internal_gp;
    other.internal_gp = tmp;
    return *this;
  }

  /*
   ***************************************************************************
   *
   * GAM primitives
   *
   ***************************************************************************
   */

  /**
   * @brief disruptively transforms a private pointer (a.k.a. parent)
   * into a local reference (a.k.a. child)
   *
   * local returns a local reference to the memory pointed by a global pointer
   * and returns it as a pointer of type \ref gam_unique_ptr. The input private
   * pointer is destroyed (otherwise could be dangling).
   */
  gam_unique_ptr<T> local() {
    if (internal_gp.is_address() && ctx().am_owner(internal_gp)) {
      T *lp = ctx().local_private<T>(internal_gp);

      /* neutralize parent destruction */
      release();

      /* prepare child deleter */
      auto deleter = [](T *lp) {
        assert(ctx().has_parent(lp));
        ctx().unmap(ctx().parent(lp));
      };

      return gam_unique_ptr<T>(lp, deleter);
    }

    if (!internal_gp.is_address()) {
      std::cerr << "> called local() for non-address pointer:\n"
                << internal_gp << std::endl;
    }
    if (!ctx().am_owner(internal_gp)) {
      std::cerr << "> called local() for non-owned pointer:\n"
                << internal_gp << std::endl;
    }
    return gam_unique_ptr<T>(nullptr, [](T *) {});
  }

  /**
   * @ brief disruptively transfers a private pointer to another executor
   *
   * @param to is the executor to transfer to
   */
  void push(executor_id to) {
    if (to != ctx().rank() && to < ctx().cardinality()) {
      if (internal_gp.is_address()) {
        // pointer brings a global address
        if (ctx().am_owner(internal_gp)) {
          ctx().push_private(internal_gp, to);
          release();
        } else
          std::cerr << "> called push() for non-owned pointer:\n"
                    << internal_gp << std::endl;
      }

      else {
        // pointer brings a reserved value
        ctx().push_reserved(internal_gp, to);
      }
    } else
      std::cerr << "> called push() towards invalid rank: " << to << std::endl;
  }

  /*
   ***************************************************************************
   *
   * reset, release and get have the same semantics as in std::unique_ptr
   *
   ***************************************************************************
   */
  void release() noexcept { internal_gp.address(0); }

  void reset() noexcept {
    LOGLN_OS("PVT reset=" << *this);

    if (ctx().author(internal_gp) == ctx().rank())
      ctx().unmap(internal_gp);
    else
      ctx().forward_reset(internal_gp, ctx().author(internal_gp));

    release();
  }

  GlobalPointer get() const noexcept { return internal_gp; }

  /**
   * @brief pretty-prints the pointer
   */
  friend std::ostream &operator<<(std::ostream &out, const private_ptr &f) {
    return out << "[PVT global=" << f.internal_gp << "]";
  }

  /*
   ***************************************************************************
   *
   * nullptr creation, comparison and assignment
   *
   ***************************************************************************
   */
  private_ptr(std::nullptr_t) noexcept {}

  private_ptr &operator=(std::nullptr_t) {
    if (internal_gp.is_address()) {
      LOGLN_OS("PVT nullptr assignment sub=" << *this);
      reset();
    }

    return *this;
  }

  bool operator==(std::nullptr_t) noexcept {
    return internal_gp.address() == 0;
  }

  friend bool operator==(std::nullptr_t, const private_ptr &__x) noexcept {
    return __x == nullptr;
  }

  bool operator!=(std::nullptr_t) noexcept {
    return internal_gp.address() != 0;
  }

  friend bool operator!=(std::nullptr_t, const private_ptr &__x) noexcept {
    return __x != nullptr;
  }

 private:
  GlobalPointer internal_gp;

  template <typename Deleter>
  bool make(T *lp, Deleter d) {
    internal_gp = ctx().mmap_private(*lp, d);
    return internal_gp.is_address();
  }

  template <typename _Tp>
  bool writeback(gam_unique_ptr<_Tp> &&child) {
    LOGLN_OS("PVT writeback unique=" << child.get());

    _Tp *lp = child.get();

    if (ctx().has_parent(lp) && ctx().am_owner(ctx().parent(lp))) {
      internal_gp = ctx().parent(lp);
      return true;
    }

    return false;
  }
};

template <typename _Tp, typename... _Args>
private_ptr<_Tp> make_private(_Args &&... __args) {
  return private_ptr<_Tp>(                       //
      NEW<_Tp>(std::forward<_Args>(__args)...),  //
      DELETE<_Tp>);
}

/**
 * @ brief blocking pull for an incoming private pointer from another executor
 *
 * @param from is the executor to pull from
 * @retval the incoming pointer
 */
template <typename T>
private_ptr<T> pull_private(executor_id from) {
  if (from != ctx().rank() && from < ctx().cardinality())
    return private_ptr<T>(ctx().pull_private(from));
  std::cerr << "> pull_private() towards invalid rank: " << from << std::endl;
  return nullptr;
}

/**
 * @ brief blocking pull for an incoming private pointer from any other executor
 *
 * @retval the incoming pointer
 */
template <typename T>
private_ptr<T> pull_private() noexcept {
  return private_ptr<T>(ctx().pull_private());
}

} /* namespace gam */

#endif /* INCLUDE_GAM_PRIVATE_PTR_HPP_ */
