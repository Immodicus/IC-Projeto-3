#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <string.h>
#include "BitSet.h"
#include "BitStream.h"

#define BREAK_ALL(expr) if(expr) goto end;

class GolombCoder
{
private:
    static bool GetBit(uint64_t c, uint64_t pos)
    {
        return (c >> (63 - (pos % 64))) & 1;
    }

    static void SetBit(uint64_t& c, uint64_t pos, bool value)
    {
        if (value)
        {
            c |= (1 << (63 - pos));
        }
        else
        {
            c &= ~(1 << (63 - pos));
        }
    }

    static int64_t Quantize(int64_t s, uint8_t bBits)
    {
        double sample = static_cast<double>(s);
        double delta = pow(2, 16 - bBits);

        double result = delta * (std::floor(sample / delta) + 0.5);

        return static_cast<int64_t>(result);
    }


public:
    static BitSet Encode(int64_t i, uint64_t m)
    {
        bool sign = i > 0;
        
        uint64_t q = abs(i) / m;
        uint64_t r = abs(i) % m;

        uint64_t b = std::floor(log2(static_cast<double>(m)));
        
        size_t s = q + 1 + 1;

        if(r < pow(2, b+1) - m) 
        {
            s += b;
        }
        else
        {
            s += (b + 1);
        }

        BitSet bs(s);

        for(size_t k = 0; k < q; k++)
        {
            bs.SetBit(k, false);
        }
        bs.SetBit(q, true);

        if(r < pow(2, b+1) - m) // code r in binary representation using b bits
        {
            for(size_t k = q+1, h = b-1; k < (q+1)+b; k++, h--)
            {
                bs.SetBit(k, GetBit(r, 63-h));
            }
        }
        else // code the number r+2^{b+1}-M} in binary representation using b + 1 bits.
        {
            r = r + static_cast<uint64_t>(pow(2, b+1)) - m;

            for(size_t k = q+1, h = b; k < (q+1)+b+1; k++, h--)
            {
                bs.SetBit(k, GetBit(r, 63-h));
            }
        }

        bs.SetBit(s-1, sign);

        return bs;
    }

    static int64_t Decode(const BitSet& bs, uint64_t m)
    {
        uint64_t b = std::floor(log2(static_cast<double>(m)));
        
        uint64_t q = 0;

        for(; q < bs.size(); q++)
        {
            if(bs[q]) break;
        }

        uint64_t r = 0;

        for(size_t i = 0; i < b; i++)
        {
            SetBit(r, 63-b+i+1, bs[q+1+i]);
        }

        if(r >= pow(2, b+1) - m)
        {
            r <<= 1;
            SetBit(r, 63, bs[bs.size()-2]);
            r = r - pow(2, b+1) + m;
        }

        if(!bs[bs.size()-1]) return -(q * m + r);

        return q * m + r;
    }

    static std::vector<int64_t> Decode(BitStream& bs, uint64_t m)
    {
        std::vector<int64_t> decoded(0);

        uint64_t b = std::floor(log2(static_cast<double>(m)));
        
        while(!bs.End())
        {
            bool bit = false;
            
            uint64_t q = 0;

            for(; q < UINT64_MAX; q++)
            {
                BREAK_ALL(!bs.ReadBit(bit));
                if(bit) break;
            }

            uint64_t r = 0;

            for(size_t i = 0; i < b; i++)
            {
                BREAK_ALL(!bs.ReadBit(bit));
                SetBit(r, 63-b+i+1, bit);
            }

            if(r >= pow(2, b+1) - m)
            {
                r <<= 1;
                BREAK_ALL(!bs.ReadBit(bit));
                SetBit(r, 63, bit);
                r = r - pow(2, b+1) + m;
            }

            bs.ReadBit(bit);
            if(!bit) decoded.push_back(-(q * m + r));
            else decoded.push_back(q * m + r);
        }

    end:
        return decoded;
    }
};