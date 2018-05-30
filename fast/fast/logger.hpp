/*
 * logging.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_LOGGING_HPP_
#define INCLUDE_FAST_LOGGING_HPP_

enum TLogLevel {ERROR, INFO, DEBUG}

#if defined LEVEL_DEBUG
TLogLevel logLevel = DEBUG;
#elif defined LEVEL_INFO
TLogLevel logLevel = INFO;
#else
TLogLevel logLevel = ERROR;
#endif

namespace FAST {
class Logger;
}

class FAST::Logger
{
    public:
        static Logger* getLogger();
    protected:
        Logger();
    private:
        static Logger* instance;
};


#endif /* INCLUDE_FAST_LOGGING_HPP_ */
