#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <exception>
#include <memory>
#include <vector>
#include <stdint.h>
#include <fstream>

class BitSet
{
private:
    std::vector<int8_t> vData;
    size_t iBits;

    bool GetBit(int8_t c, uint8_t pos) const
    {
        return (c >> (7 - (pos % 8))) & 1;
    }

    void SetBit(int8_t& c, uint8_t pos, bool value)
    {
        if (value)
        {
            c |= (1 << (7 - pos));
        }
        else
        {
            c &= ~(1 << (7 - pos));
        }
    }

public:
    BitSet(size_t bits) : vData(bits % 8 != 0 ? (bits / 8) + 1 : bits / 8), iBits(bits) {}
    BitSet(const std::initializer_list<int8_t>& alloc) : vData(alloc), iBits(vData.size() * 8) {}

    const int8_t* data() const
    {
        return vData.data();
    }

    int8_t* data()
    {
        return vData.data();
    }

    size_t size() const
    {
        return iBits;
    }

    size_t size()
    {
        return iBits;
    }

    void SetBit(uint64_t i, bool value)
    {
        SetBit(vData[i / 8], i % 8, value);
    }

    bool GetBit(uint64_t i)
    {
        return GetBit(vData[i / 8], i % 8);
    }

    bool GetBit(uint64_t i) const
    {
        return GetBit(vData[i / 8], i % 8);
    }

    bool operator[](uint64_t i) const
    {
        if (i >= iBits) throw new std::runtime_error("Invalid index");

        return GetBit(vData[i / 8], i % 8);
    }

    friend std::ostream& operator << (std::ostream& os, const BitSet& bs)
    {
        for(size_t i = 0; i < bs.size(); i++)
        {
            os << bs[i];
        }
        os << "\n";

        return os;
    }
};