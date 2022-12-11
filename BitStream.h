#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <exception>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <string.h>
#include "BitSet.h"

class BitStream
{
private:
	std::vector<int8_t> vBuffer;

	FILE* fPtr;

	const uint64_t iMaxBufferSize = 32768;

	uint64_t iCurBufferSize;
	uint64_t iCurBit;
	uint64_t iFileLength;
	uint64_t iBufferStart;
	int64_t iLastWrittenPos;

	bool bExtended;

	bool GetBit(int8_t b, int8_t pos)
	{
		return (b >> (7 - pos)) & 1;
	}

	void SetBit(int8_t& b, int8_t pos, bool value)
	{
		if (value)
		{
			b |= (1 << (7 - pos));
		}
		else
		{
			b &= ~(1 << (7 - pos));
		}
	}

	bool IsBitCached(uint64_t bit)
	{
		bit /= 8;
		return bit >= iBufferStart && bit < iBufferStart + iCurBufferSize;
	}

	bool IsBitOutside(uint64_t bit)
	{
		bit /= 8;
		return bit > iFileLength - 1;
	}

	uint64_t GetFileSize()
	{
		fseek(fPtr, 0, SEEK_END);
		return ftell(fPtr);
	}

	void Fill(uint64_t pos = 0)
	{
		pos /= 8;
		Flush();
		iBufferStart = pos;
		fseek(fPtr, iBufferStart, SEEK_SET);
		iCurBufferSize = fread(vBuffer.data(), sizeof(int8_t), vBuffer.size(), fPtr);
		iLastWrittenPos = -1;
		bExtended = false;
	}

	void Extend(uint64_t pos = 0)
	{
		pos /= 8;
		Flush();
		iBufferStart = pos;
		iCurBufferSize = iMaxBufferSize;
		iFileLength = iBufferStart + iMaxBufferSize;
		iLastWrittenPos = -1;
		memset(vBuffer.data(), 0, vBuffer.size());
		bExtended = true;
		if (iCurBit % 8 != 0)
		{
			fseek(fPtr, iBufferStart, SEEK_SET);
			fread(vBuffer.data(), sizeof(int8_t), 1, fPtr);
		}
	}

	void Flush()
	{
		if (iCurBufferSize > 0 && iLastWrittenPos >= 0)
		{
			fseek(fPtr, iBufferStart, SEEK_SET);
			fwrite(vBuffer.data(), sizeof(int8_t), iLastWrittenPos + 1, fPtr);
			if (bExtended)
			{
				iFileLength = (iBufferStart + iLastWrittenPos + 1);
			}
		}
	}

	void Close()
	{
		fclose(fPtr);
	}

public:
	BitStream(const char* fileName, const char* mode) : vBuffer((uint32_t)iMaxBufferSize), fPtr(NULL), iCurBufferSize(0), iCurBit(0), iFileLength(0), iBufferStart(0), iLastWrittenPos(-1), bExtended(false)
	{
		vBuffer.resize((uint32_t)iMaxBufferSize);
		fPtr = fopen(fileName, mode);
		setvbuf(fPtr, NULL, _IONBF, 0); // disable buffer since we have our own

		if (!fPtr) 
		{
			throw std::runtime_error("Failed to open file");
		}

		if (!(iFileLength = GetFileSize()))
		{
			Extend();
		}
		else
		{
			Fill();
		}
	}

	~BitStream()
	{
		Flush();
		Close();
	}

	uint64_t CurrentBit()
	{
		return iCurBit;
	}

	void ReadAlign()
	{
		bool t;
		
		while(iCurBit % 8 != 0) assert(ReadBit(t));
	}

	void WriteAlign(bool v)
	{		
		while(iCurBit % 8 != 0) WriteBit(v);
	}

	bool End()
	{
		if (!IsBitCached(iCurBit))
		{
			if (IsBitOutside(iCurBit)) return true;
		}

		return false;
	}

	bool ReadBit(bool& b)
	{
		if (!IsBitCached(iCurBit))
		{
			if (IsBitOutside(iCurBit)) return false;
			else Fill(iCurBit);
		}

		b = GetBit(vBuffer[((iCurBit / 8) - iBufferStart)], iCurBit % 8);

		iCurBit++;

		return true;
	}

	BitSet ReadNBits(uint64_t nBits)
	{
		if (nBits == 0) return BitSet(0);
		if (nBits > iMaxBufferSize * 8) return BitSet(0);

		if (!IsBitCached(iCurBit) || !IsBitCached(iCurBit + nBits - 1))
		{
			if (IsBitOutside(iCurBit))
			{
				return BitSet(0);
			}
			else if (IsBitOutside(iCurBit + nBits - 1))
			{
				nBits = (iFileLength) * 8 - iCurBit;
			}
			
			Fill(iCurBit);
		}

		BitSet b(nBits);
		if (iCurBit % 8 == 0)
		{
			memcpy(b.data(), vBuffer.data() + ((iCurBit / 8) - iBufferStart), nBits % 8 != 0 ? (nBits / 8) + 1 : nBits / 8);
			iCurBit += nBits;
		}
		else
		{
			for (uint64_t i = 0; i < nBits; i++, iCurBit++)
			{
				b.SetBit(i, GetBit(vBuffer[((iCurBit / 8) - iBufferStart)], iCurBit % 8));
			}
		}

		return b;
	}

	void WriteBit(bool value)
	{
		if (!IsBitCached(iCurBit))
		{
			if (IsBitOutside(iCurBit))
			{
				Extend(iCurBit);
			}
			else
			{
				Fill(iCurBit);
			}
		}

		SetBit(vBuffer[((iCurBit / 8) - iBufferStart)], iCurBit % 8, value);

		if ((int64_t)((iCurBit / 8) - iBufferStart) > iLastWrittenPos) iLastWrittenPos = (int64_t)((iCurBit / 8) - iBufferStart);

		iCurBit++;
	}

	bool WriteNBits(const BitSet& bits)
	{
		if (bits.size() == 0 || bits.size() > iMaxBufferSize * 8) return false;

		if (!IsBitCached(iCurBit) || !IsBitCached(iCurBit + bits.size() - 1))
		{
			if (IsBitOutside(iCurBit))
			{
				Extend(iCurBit);
			}
			else if (IsBitOutside(iCurBit + bits.size() - 1))
			{
				Extend(iCurBit);
			}
			else
			{
				Fill(iCurBit);
			}
		}

		if (iCurBit % 8 == 0 && bits.size() % 8 == 0)
		{
			memcpy(vBuffer.data() + ((iCurBit / 8) - iBufferStart), bits.data(), bits.size() / 8);
			iCurBit += bits.size();
		}
		else
		{
			for (uint64_t i = 0; i < bits.size(); i++, iCurBit++)
			{
				SetBit(vBuffer[((iCurBit / 8) - iBufferStart)], iCurBit % 8, bits[i]);
			}
		}

		if ((int64_t)((((iCurBit-1) / 8)) - iBufferStart) > iLastWrittenPos) iLastWrittenPos = (int64_t)((((iCurBit - 1) / 8)) - iBufferStart);

		return true;
	}

	template<typename T>
	bool Write(const T& v)
	{
		if(iCurBit % 8 != 0) return false;

		constexpr size_t sSize = sizeof(T);
		constexpr size_t sBits = sSize * 8;

		if (!IsBitCached(iCurBit) || !IsBitCached(iCurBit + sBits - 1))
		{
			Extend(iCurBit);
		}

		memcpy(vBuffer.data() + ((iCurBit / 8) - iBufferStart), &v, sSize);

		iCurBit += sBits;

		if ((int64_t)((((iCurBit-1) / 8)) - iBufferStart) > iLastWrittenPos) iLastWrittenPos = (int64_t)((((iCurBit - 1) / 8)) - iBufferStart);

		return true;
	}

	template<typename T>
	bool Read(T& v)
	{
		if(iCurBit % 8 != 0) return false;

		constexpr size_t sSize = sizeof(T);
		constexpr size_t sBits = sSize * 8;

		if (!IsBitCached(iCurBit) || !IsBitCached(iCurBit + sBits - 1))
		{
			if (IsBitOutside(iCurBit) || IsBitOutside(iCurBit + sBits - 1))
			{
				return false;
			}
			
			Fill(iCurBit);
		}

		memcpy(&v, vBuffer.data() + ((iCurBit / 8) - iBufferStart), sSize);

		iCurBit += sBits;

		return true;
	}
};