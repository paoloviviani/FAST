/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_LOGGING_HPP_
#define INCLUDE_FAST_LOGGING_HPP_

#include <fstream>

using namespace std;

enum LogLevel {ERROR, INFO, DEBUG}

#if defined LEVEL_DEBUG
LogLevel logLevel = DEBUG;
#elif defined LEVEL_INFO
LogLevel logLevel = INFO;
#else
LogLevel logLevel = ERROR;
#endif

namespace FAST {

}
#endif /* INCLUDE_FAST_LOGGING_HPP_ */
