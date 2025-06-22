#include "timegm_helper.h"
#include "esphome/core/log.h"
#include <cstdio>

namespace esphome {
namespace timegm_helper {

static const char *const TAG = "timegm_helper";

void TimegmHelperComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up TimegmHelper component");
}

std::string parse_time(const std::string &x, const char format[], time::RealTimeClock *rtc) {
    struct tm tm = {};
    strptime(x.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
    time_t utc_time = my_timegm(&tm, rtc);
    struct tm buf = {};
    struct tm* cest_tm = localtime_r(&utc_time, &buf);

    char result[64];
    size_t len = strftime(result, sizeof(result), format, cest_tm);
    if (len == 0) {
        ESP_LOGE(TAG, "Failed to format time for %s with format %s and timestamp %ld", x.c_str(), format, utc_time);
        return "";
    }

    return std::string(result, len);
}

time_t my_timegm(struct tm *tm, time::RealTimeClock *rtc)
{
    time_t ret;
    char *tz;

    // Set timezone to UTC
    setenv("TZ", "UTC0", 1);
    tzset();

    // Convert to time_t
    ret = mktime(tm);

    // Restore original timezone
    setenv("TZ", rtc->get_timezone().c_str(), 1);
    tzset();

    return ret;
}

}  // namespace timegm_helper
}  // namespace esphome
