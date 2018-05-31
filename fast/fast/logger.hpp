/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_LOGGING_HPP_
#define INCLUDE_FAST_LOGGING_HPP_

using namespace std;

#if defined LOGLEVEL_DEBUG
auto logLevel = plog::debug;
#elif defined LOGLEVEL_INFO
auto logLevel = plog::info;
#elif defined LOGLEVEL_ERROR
auto logLevel = plog::error;
#else
auto logLevel = plog::info;
#endif

static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
plog::init(plog::verbose, &consoleAppender);

#endif /* INCLUDE_FAST_LOGGING_HPP_ */
