//
// Created by ksztn on 14/10/2026.
//

#ifndef CLUA_CHAR_SET_H
#define CLUA_CHAR_SET_H

#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h>

class RuntimeCharSet
{
private:
    alignas(32) std::array<std::uint8_t, 32> data{};

public:
    constexpr RuntimeCharSet() = default;

    constexpr explicit RuntimeCharSet(
        const std::array<std::uint8_t, 32>& assign_data)
        : data(assign_data)
    {
    }

    void insert(std::uint8_t ch)
    {
        data[ch >> 3] |=
            static_cast<std::uint8_t>(1u << (ch & 7));
    }

    void remove(std::uint8_t ch)
    {
        data[ch >> 3] &=
            static_cast<std::uint8_t>(~(1u << (ch & 7)));
    }

    bool contains(std::uint8_t ch) const
    {
        return
            (data[ch >> 3] &
             (1u << (ch & 7))) != 0;
    }

    void insert_range(
        std::uint8_t char_a,
        std::uint8_t char_b)
    {
        if (char_a > char_b)
        {
            std::swap(char_a, char_b);
        }

        for (std::uint32_t i = char_a;
             i <= char_b;
             ++i)
        {
            insert(static_cast<std::uint8_t>(i));
        }
    }

    bool is_empty() const
    {
        const auto vec =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    data.data()));

        return _mm256_testz_si256(vec, vec);
    }

    bool has_overlap(
        const RuntimeCharSet& other) const
    {
        const auto lhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    data.data()));

        const auto rhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    other.data.data()));

        const auto overlap =
            _mm256_and_si256(lhs, rhs);

        return !_mm256_testz_si256(
            overlap,
            overlap);
    }

    RuntimeCharSet sum(
        const RuntimeCharSet& other) const
    {
        RuntimeCharSet result;

        const auto lhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    data.data()));

        const auto rhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    other.data.data()));

        const auto res =
            _mm256_or_si256(lhs, rhs);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                result.data.data()),
            res);

        return result;
    }

    RuntimeCharSet overlap(
        const RuntimeCharSet& other) const
    {
        RuntimeCharSet result;

        const auto lhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    data.data()));

        const auto rhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    other.data.data()));

        const auto res =
            _mm256_and_si256(lhs, rhs);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                result.data.data()),
            res);

        return result;
    }

    RuntimeCharSet negate(
        const RuntimeCharSet& other) const
    {
        RuntimeCharSet result;

        const auto lhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    data.data()));

        const auto rhs =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    other.data.data()));

        const auto res =
            _mm256_andnot_si256(
                rhs,
                lhs);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                result.data.data()),
            res);

        return result;
    }

    RuntimeCharSet operator|(
        const RuntimeCharSet& other) const
    {
        return sum(other);
    }

    RuntimeCharSet operator&(
        const RuntimeCharSet& other) const
    {
        return overlap(other);
    }

    RuntimeCharSet operator-(
        const RuntimeCharSet& other) const
    {
        return negate(other);
    }
};

#endif //CLUA_CHAR_SET_H
