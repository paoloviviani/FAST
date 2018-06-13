/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_LOGGING_HPP_
#define FAST_FAST_LOGGING_HPP_

#include <fstream>
#include <iostream>
#include <cstdarg>
#include <string>
#include <unistd.h>
#include <mutex>
#include <cassert>
#include "gam.hpp"

using namespace std;

/*
 * error reporting facilities
 */
#define FASTASSERT(x) assert(x)

#if defined LOGLEVEL_DEBUG
#define FAST_DEBUG(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_output() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}
#define FAST_INFO(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_output() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}
#define FAST_ERROR(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_error() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}

#elif defined LOGLEVEL_INFO
#define FAST_DEBUG(...) {}
#define FAST_INFO(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_output() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}
#define FAST_ERROR(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_error() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}

#else
#define FAST_DEBUG(...) {}
#define FAST_INFO(...) {}
#define FAST_ERROR(x) {\
		FAST::Logger::getLogger()->lock(); \
		FAST::Logger::getLogger()->log_error() << x << std::endl; \
		FAST::Logger::getLogger()->unlock();}

#endif

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
inline const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
//	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
	strftime(buf, sizeof(buf), "%X", &tstruct);
	return buf;
}

/*
 * Pretty printing for std::vector
 */
template < class T >
inline std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (auto item : v)
    {
        os << item << ", ";
    }
    os << " ]";
    return os;
}

namespace FAST {

class Logger {
public:
	static Logger *getLogger() {
		static Logger logger;
		return &logger;
	}

	void init() {
		id = gam::rank();
		log("I am FAST worker %d (pid=%d)", id);
	}

	void finalize() {
		//print footer message
		log("stop logging worker %d", id);
	}

	/**
	 * Standard error stream provider
	 * @return
	 */
	std::ostream &log_error() {
		return cout << "[" << currentDateTime() << ", proc " << id <<"] ";
	}

	/**
	 * Standard output stream provider
	 * @return
	 */
	std::ostream &log_output() {
		return cerr << "[" << currentDateTime() << ", proc " << id <<"] ";
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
	uint32_t id;
	std::mutex mtx;

	char sMessage[256];
	va_list args;

};

}

#endif /* FAST_FAST_LOGGING_HPP_ */
