/**
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Alex Popescu (@pspsalex)
 */

#include "timegm_helper.h"
#include "esphome/components/time/posix_tz.h"
#include "esphome/core/log.h"
#include <cstdio>
#include <cinttypes>

namespace esphome {
namespace timegm_helper {

static const char *const TAG = "timegm_helper";

std::string parse_time(const std::string &x, const char format[], time::RealTimeClock *rtc) {
    struct tm tm = {};
    strptime(x.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
    time_t utc_time = my_timegm(&tm, rtc);
    struct tm buf = {};
    struct tm* cest_tm = localtime_r(&utc_time, &buf);

    char result[64];
    size_t len = strftime(result, sizeof(result), format, cest_tm);
    if (len == 0) {
        ESP_LOGE(TAG, "Failed to format time for %s with format %s and timestamp %" PRIu64, x.c_str(), format, static_cast<uint64_t>(utc_time));
        return "";
    }

    return std::string(result, len);
}

time_t my_timegm(struct tm *tm, time::RealTimeClock *rtc)
{
    time_t ret;
    auto tz = time::get_global_tz();

    // Set timezone to UTC
    rtc->set_timezone(nullptr);

    // Convert to time_t
    ret = mktime(tm);

    // Restore original timezone
    time::set_global_tz(tz);

    return ret;
}

}  // namespace timegm_helper
}  // namespace esphome
