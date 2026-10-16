#pragma once

#include <stdint.h>

#include "driver/gpio.h"
#include "driver/mcpwm_prelude.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

namespace learm {

enum class ServoChannel : uint8_t {
    S0 = 0,
    S1,
    S2,
    S3,
    S4,
    S5,
};

inline constexpr uint8_t kServoCount = 6;

inline constexpr uint32_t kServoFrequencyHz = 50;
inline constexpr uint16_t kServoPeriodUs = 20000;

inline constexpr uint16_t kServoMinUs = 500;
inline constexpr uint16_t kServoMaxUs = 2500;

struct ServoPwmConfig {
    ServoChannel channel;
    gpio_num_t gpio;

    uint16_t reset_us;
    uint16_t min_us;
    uint16_t max_us;

    int16_t offset_us;
};

struct ServoRuntimeState {
    bool configured;
    bool enabled;
    uint16_t command_us;
    uint16_t output_us;
    uint16_t reset_us;
    uint16_t min_us;
    uint16_t max_us;
    int16_t offset_us;
};

class ServoPwm {
public:
    esp_err_t init(const ServoPwmConfig* configs, uint8_t count);
    esp_err_t deinit();

    esp_err_t enable(ServoChannel channel, uint16_t start_us);
    esp_err_t disable(ServoChannel channel);
    esp_err_t disable_all();
    bool is_enabled(ServoChannel channel) const;

    esp_err_t write_us_now(ServoChannel channel, uint16_t pulse_us);

    esp_err_t set_config(ServoChannel channel, const ServoPwmConfig& config);
    ServoPwmConfig get_config(ServoChannel channel) const;

    esp_err_t set_offset(ServoChannel channel, int16_t offset_us);
    int16_t get_offset(ServoChannel channel) const;

    uint16_t read_command_us(ServoChannel channel) const;
    uint16_t read_output_us(ServoChannel channel) const;

    ServoRuntimeState get_state(ServoChannel channel) const;

private:
    struct ServoState {
        bool configured = false;
        bool enabled = false;
        ServoPwmConfig config = {};
        uint16_t command_us = 1500;
        uint16_t output_us = 1500;
    };

    class LockGuard {
    public:
        explicit LockGuard(const ServoPwm* owner, TickType_t timeout_ticks = portMAX_DELAY);
        ~LockGuard();
        bool locked() const;

    private:
        const ServoPwm* owner_ = nullptr;
        bool locked_ = false;
    };

    bool take_lock(TickType_t timeout_ticks = portMAX_DELAY) const;
    void give_lock() const;

    esp_err_t setup_pwm_backend();
    esp_err_t teardown_pwm_backend();
    esp_err_t create_channel_resources(uint8_t index, uint16_t initial_output_us);
    esp_err_t destroy_channel_resources(uint8_t index);
    esp_err_t write_output_us(uint8_t index, uint16_t output_us);

    uint16_t clamp_command_us(uint8_t index, uint16_t pulse_us) const;
    uint16_t apply_offset_and_clamp(uint8_t index, uint16_t command_us) const;

    static int channel_to_index(ServoChannel channel);

private:
    bool initialized_ = false;
    SemaphoreHandle_t lock_ = nullptr;
    ServoState states_[kServoCount] = {};
    static constexpr uint8_t kMcpwmOperatorCount = 3;
    mcpwm_timer_handle_t mcpwm_timer_ = nullptr;
    mcpwm_oper_handle_t mcpwm_operators_[kMcpwmOperatorCount] = {};
    mcpwm_cmpr_handle_t mcpwm_comparators_[kServoCount] = {};
    mcpwm_gen_handle_t mcpwm_generators_[kServoCount] = {};
};

}  // namespace learm
