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

#define BREAK_ONE(expr) if(expr) return INT64_MIN

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
    template <typename T>
    static int64_t ComputeRequiredBits(const std::vector<T> &v, int64_t t, bool fold)
    {
        uint64_t b = std::floor(log2(static_cast<double>(t)));
        uint64_t sum = 0;

        for (auto e : v)
        {
            if(fold) 
            {
                if(e < 0) e = abs(e * 2) - 1;
                else e = abs(e * 2);
            }
            
            uint64_t q = abs(e) / t;
            uint64_t r = abs(e) % t;

            size_t s = q + 1;
            if(!fold) s++;

            if (r < pow(2, b + 1) - t)
            {
                s += b;
            }
            else
            {
                s += (b + 1);
            }

            sum += s;
        }

        return sum;
    }

    template<typename T>
    static int64_t EstimateM(const std::vector<T>& v, int64_t range, int64_t div, bool fold = false)
    {
        static_assert(std::is_fundamental<T>::value);
        
        uint64_t sum = 0;
        for(const auto e : v)
        {
            sum += abs(e);
        }

        int64_t m = std::ceil(sum / v.size());

        uint64_t min = UINT64_MAX;
        int64_t bestM = m;

        while(range > 1 && div > 1)
        {
            m = bestM;
            
            std::vector<uint64_t> mins;

            for(int64_t t = bestM - range; t <= bestM + range; t += div)
            {
                if(t <= 0) mins.push_back(UINT64_MAX);
                
                else mins.push_back(ComputeRequiredBits(v, t, fold));
            }

            for(size_t i = 0; i < mins.size(); i++)
            {
                if(mins[i] < min)
                {
                    min = mins[i];
                    bestM = (m - range) + i * div;
                }
            }

            range = 2 * div;
            div /= 2;
        }

        return bestM;
    }

    template<typename T>
    static int64_t EstimateMBrute(const std::vector<T>& v, int64_t range, bool fold = false)
    {
        static_assert(std::is_fundamental<T>::value);
        
        uint64_t sum = 0;
        for(const auto e : v)
        {
            sum += abs(e);
        }

        int64_t m = std::ceil(sum / v.size());

        uint64_t min = UINT64_MAX;
        int64_t bestM = m;

        for(int64_t t = bestM - range; t <= bestM + range; t++)
        {
            if(t <= 0) continue;

            uint64_t e = ComputeRequiredBits(v, t, fold); 
            if(e < min)
            {
                min = e;
                bestM = t;
            }
        }

        return bestM;
    }

    template<typename T>
    static int64_t EstimateMFast(const std::vector<T>& v)
    {
        static_assert(std::is_fundamental<T>::value);
        
        uint64_t sum = 0;
        for(const auto e : v)
        {
            sum += abs(e);
        }

        int64_t m = std::ceil(sum / v.size());

        return m;
    }

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

    static std::vector<int64_t> Decode(BitStream& bs, uint64_t m, uint64_t max)
    {
        std::vector<int64_t> decoded(0);

        uint64_t b = std::floor(log2(static_cast<double>(m)));

        uint64_t count = 0;
        
        while(!bs.End() && count < max)
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

            count++;
        }

    end:
        return decoded;
    }

    static BitSet EncodeFold(int64_t i, uint64_t m)
    {     
        i *= 2;
        if(i < 0) i = abs(i) - 1;
        else i = abs(i);
        
        uint64_t q = abs(i) / m;
        uint64_t r = abs(i) % m;

        uint64_t b = std::floor(log2(static_cast<double>(m)));
        
        size_t s = q + 1;

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

        return bs;
    }

    static int64_t DecodeFold(const BitSet& bs, uint64_t m)
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
            SetBit(r, 63, bs[bs.size()-1]);
            r = r - pow(2, b+1) + m;
        }

        int64_t res = q * m + r;
        if(res % 2 == 1) res = -(res / 2) - 1;
        else res = res / 2;

        return res;
    }

    static std::vector<int64_t> DecodeFold(BitStream& bs, uint64_t m, uint64_t max = UINT64_MAX)
    {
        std::vector<int64_t> decoded(0);

        uint64_t b = std::floor(log2(static_cast<double>(m)));

        uint64_t count = 0;
        
        while(!bs.End() && count < max)
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

            int64_t res = q * m + r;
            if(res % 2 == 1) res = -(res / 2) - 1;
            else res = res / 2;

            decoded.push_back(res);
            count++;
        }

    end:
        return decoded;
    }

    inline static int64_t DecodeOneFold(BitStream& bs, uint64_t m)
    {
        uint64_t b = std::floor(log2(static_cast<double>(m)));
        
        bool bit = false;

        uint64_t q = 0;

        for (; q < UINT64_MAX; q++)
        {
            BREAK_ONE(!bs.ReadBit(bit));
            if (bit)
                break;
        }

        uint64_t r = 0;

        for (size_t i = 0; i < b; i++)
        {
            BREAK_ONE(!bs.ReadBit(bit));
            SetBit(r, 63 - b + i + 1, bit);
        }

        if (r >= pow(2, b + 1) - m)
        {
            r <<= 1;
            BREAK_ONE(!bs.ReadBit(bit));
            SetBit(r, 63, bit);
            r = r - pow(2, b + 1) + m;
        }

        int64_t res = q * m + r;
        if (res % 2 == 1)
            res = -(res / 2) - 1;
        else
            res = res / 2;

        return res;
    }
};