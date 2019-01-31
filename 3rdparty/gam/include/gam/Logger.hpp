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
 * @brief       implements GAM Logger class and shortcut macros
 *
 * @ingroup     internals
 */
#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <unistd.h>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

/*
 * shortcuts
 */
#ifdef GAM_LOG
#define LOGLN (gam::Logger::getLogger()->log)
#define LOGLN_OS(x)                                           \
  {                                                           \
    gam::Logger::getLogger()->lock();                         \
    gam::Logger::getLogger()->out_stream() << x << std::endl; \
    gam::Logger::getLogger()->unlock();                       \
  }
#define LOGGER_INIT (gam::Logger::getLogger()->init)
#define LOGGER_FINALIZE (gam::Logger::getLogger()->finalize)
#else
#define LOGLN(...) \
  {}
#define LOGLN_OS(...) \
  {}
#define LOGGER_INIT(...) \
  {}
#define LOGGER_FINALIZE(...) \
  {}
#endif

namespace gam {

class Logger {
 public:
  static Logger *getLogger() {
    static Logger logger;
    return &logger;
  }

  void init(int id) {
    // print header message
    log("I am gam executor %d (pid=%d)", id, getpid());
  }
  void finalize(int id) {
    // print footer message
    log("stop logging executor %d", id);
  }

  /**
   *   Variable Length Logger function
   *   @param format string for the message to be logged.
   */
  void log(const char *format, ...) {
    // print message
    va_start(args, format);
    vsprintf(sMessage, format, args);

    lock();
    std::cout << "[" << time(0) << "] " << sMessage << std::endl;
    unlock();

    va_end(args);
  }

  std::ostream &out_stream() { return std::cout << "[" << time(0) << "] "; }

  void lock() { mtx.lock(); }

  void unlock() { mtx.unlock(); }

 private:
  std::mutex mtx;

  char sMessage[256];
  va_list args;
};

} /* namespace gam */

#endif /* SKEDATO_LOGGER_HPP_ */
