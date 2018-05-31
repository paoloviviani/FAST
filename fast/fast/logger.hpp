/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_LOGGING_HPP_
#define INCLUDE_FAST_LOGGING_HPP_

#include <fstream>
#include <iostream>
#include <cstdarg>
#include <string>
#include <unistd.h>
#include <mutex>
#include "fast/gam_wrapper.hpp"

using namespace std;


#if defined LOGLEVEL_DEBUG
#define FAST_DEBUG(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_output() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}
#define FAST_INFO(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_output() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}
#define FAST_ERROR(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_error() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}

#elif defined LOGLEVEL_INFO
#define FAST_DEBUG(...) {}
#define FAST_INFO(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_output() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}
#define FAST_ERROR(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_error() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}

#else
#define FAST_DEBUG(...) {}
#define FAST_INFO(...) {}
#define FAST_ERROR(x) {\
    gam::Logger::getLogger()->lock(); \
    gam::Logger::getLogger()->log_error() << x << std::endl; \
    gam::Logger::getLogger()->unlock();}

#endif


namespace FAST {

class Logger {
public:
	static Logger *getLogger() {
		static Logger logger;
		return &logger;
	}

	void init() {
		uint32_t id = FAST::rank();
		log("I am FAST worker %d (pid=%d)", id, getpid());
	}

	void finalize(int id = 0) {
		//print footer message
		log("stop logging worker %d", id);
	}

	/**
	 * Standard error stream provider
	 * @return
	 */
	std::ostream &log_error() {
			return cout << "[" << time(0) << "] ";
	}
	/**
	 * Standard output stream provider
	 * @return
	 */
	std::ostream &log_output() {
		return cerr << "[" << time(0) << "] ";
	}

	/**
	 *   Variable Length Logger function
	 *   @param format string for the message to be logged.
	 */
	void log(const char * format, ...) {
		//print message
		va_start(args, format);
		vsprintf(sMessage, format, args);
		lock();
		log_output() << sMessage << std::endl;
		unlock();
		va_end(args);
	}

	void lock() {
		mtx.lock();
	}

	void unlock() {
		mtx.unlock();
	}

private:
	std::mutex mtx;

	char sMessage[256];
	va_list args;

};

}

#endif /* INCLUDE_FAST_LOGGING_HPP_ */
