/**
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Alex Popescu (@pspsalex)
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/components/time/real_time_clock.h"
#include <time.h>
#include <cstdlib>

namespace esphome {
namespace timegm_helper {

/**
 * Custom timegm implementation for ESPHome
 * Converts a struct tm to time_t assuming UTC timezone
 *
 * @param tm Pointer to struct tm containing the time to convert
 * @return time_t timestamp in seconds since epoch
 */
time_t my_timegm(struct tm *tm,  time::RealTimeClock *rtc);

std::string parse_time(const std::string &x, const char format[], time::RealTimeClock *rtc);

class TimegmHelperComponent : public Component {
 public:
  void setup() override;

  float get_setup_priority() const override {
    return esphome::setup_priority::LATE;
  }
};

}  // namespace timegm_helper
}  // namespace esphome
