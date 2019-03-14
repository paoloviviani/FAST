/*
 * fast_utils.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_FAST_UTILS_HPP_
#define FAST_FAST_FAST_UTILS_HPP_

/*
 * Pretty printing for std::vector
 */
template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (auto item : v)
    {
        os << item << ", ";
    }
    os << " ]";
    return os;
}

/*
 * Pretty printing for FAST::gam_vector
 */
template <class T>
inline std::ostream &operator<<(std::ostream &os, const FAST::gam_vector<T> &v)
{
    os << "[";
    for (auto item : v)
    {
        os << item << ", ";
    }
    os << " ]";
    return os;
}

namespace FAST
{

constexpr auto EOI_TOKEN = gff::go_on - 1;
constexpr auto TRIGGER_TOKEN = gff::go_on - 2;
constexpr auto SYNC_TOKEN = gff::go_on - 3;

uint32_t rank()
{
    return gam::rank();
}

uint32_t cardinality()
{
    return gam::cardinality();
}

template <typename T>
bool global_sync()
{
    if (rank() != 0)
        token2public<T>(SYNC_TOKEN).push(0);
        do
        {
            auto p = gam::pull_public<T>();
        } while (p.get().address() != SYNC_TOKEN);
        
    else
    {
        int count = 0;
        while (count < cardinality() - 1)
        {
            auto p = gam::pull_public<T>();
            count++;
        }
        for (int i = 1; i < cardinality(); i++)
            token2public<T>(SYNC_TOKEN).push(i);
    }
}

} /* namespace FAST */

#endif /* FAST_FAST_FAST_UTILS_HPP_ */
