/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_LOGGING_HPP_
#define INCLUDE_FAST_LOGGING_HPP_

#include "gam/include/gam.hpp"

enum TLogLevel {INFO, DEBUG}

#ifdef DO_DEBUG
TLogLevel logLevel = DEBUG;
#else
TLogLevel logLevel = INFO;
#endif

namespace FAST {

class Logger {

};

}


#endif /* INCLUDE_FAST_LOGGING_HPP_ */
