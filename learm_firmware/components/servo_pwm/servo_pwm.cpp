#include "servo_pwm.hpp"

#include <algorithm>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

namespace learm {

static const char* TAG = "servo_pwm";
static constexpr uint32_t kMcpwmResolutionHz = 1000000;
static constexpr int kMcpwmGroupId = 0;

static uint8_t channel_to_operator_index(uint8_t index)
{
    return static_cast<uint8_t>(index / 2);
}

int ServoPwm::channel_to_index(ServoChannel channel)
{
    const uint8_t index = static_cast<uint8_t>(channel);
    return index >= kServoCount ? -1 : static_cast<int>(index);
}

uint16_t ServoPwm::clamp_command_us(uint8_t index, uint16_t pulse_us) const
{
    if (index >= kServoCount || !states_[index].configured) return pulse_us;
    const ServoPwmConfig& cfg = states_[index].config;
    return std::clamp<uint16_t>(pulse_us, cfg.min_us, cfg.max_us);
}

uint16_t ServoPwm::apply_offset_and_clamp(uint8_t index, uint16_t command_us) const
{
    if (index >= kServoCount || !states_[index].configured) return command_us;
    const ServoPwmConfig& cfg = states_[index].config;
    const int output = static_cast<int>(command_us) + static_cast<int>(cfg.offset_us);
    return static_cast<uint16_t>(std::clamp<int>(output, cfg.min_us, cfg.max_us));
}

esp_err_t ServoPwm::setup_pwm_backend()
{
    if (mcpwm_timer_ != nullptr) return ESP_OK;

    mcpwm_timer_config_t timer_config = {};
    timer_config.group_id = kMcpwmGroupId;
    timer_config.clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT;
    timer_config.resolution_hz = kMcpwmResolutionHz;
    timer_config.period_ticks = kServoPeriodUs;
    timer_config.count_mode = MCPWM_TIMER_COUNT_MODE_UP;

    esp_err_t ret = mcpwm_new_timer(&timer_config, &mcpwm_timer_);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "mcpwm_new_timer failed: %s", esp_err_to_name(ret));
        return ret;
    }

    for (uint8_t i = 0; i < kMcpwmOperatorCount; ++i) {
        mcpwm_operator_config_t operator_config = {};
        operator_config.group_id = kMcpwmGroupId;
        ret = mcpwm_new_operator(&operator_config, &mcpwm_operators_[i]);
        if (ret != ESP_OK) {
            (void)teardown_pwm_backend();
            return ret;
        }
        ret = mcpwm_operator_connect_timer(mcpwm_operators_[i], mcpwm_timer_);
        if (ret != ESP_OK) {
            (void)teardown_pwm_backend();
            return ret;
        }
    }

    ret = mcpwm_timer_enable(mcpwm_timer_);
    if (ret != ESP_OK) {
        (void)teardown_pwm_backend();
        return ret;
    }
    ret = mcpwm_timer_start_stop(mcpwm_timer_, MCPWM_TIMER_START_NO_STOP);
    if (ret != ESP_OK) {
        (void)teardown_pwm_backend();
        return ret;
    }

    ESP_LOGI(TAG, "mcpwm servo backend started: group=%d freq=%uHz period=%uus resolution=%uHz",
             kMcpwmGroupId, kServoFrequencyHz, kServoPeriodUs, kMcpwmResolutionHz);
    return ESP_OK;
}

esp_err_t ServoPwm::teardown_pwm_backend()
{
    esp_err_t first_error = ESP_OK;

    if (mcpwm_timer_ != nullptr) {
        const esp_err_t ret = mcpwm_timer_start_stop(mcpwm_timer_, MCPWM_TIMER_STOP_EMPTY);
        if (ret != ESP_OK) first_error = ret;
    }

    for (uint8_t i = 0; i < kServoCount; ++i) {
        const esp_err_t ret = destroy_channel_resources(i);
        if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    }

    for (uint8_t i = 0; i < kMcpwmOperatorCount; ++i) {
        if (mcpwm_operators_[i] != nullptr) {
            const esp_err_t ret = mcpwm_del_operator(mcpwm_operators_[i]);
            if (ret == ESP_OK) mcpwm_operators_[i] = nullptr;
            else if (first_error == ESP_OK) first_error = ret;
        }
    }

    if (mcpwm_timer_ != nullptr) {
        esp_err_t ret = mcpwm_timer_disable(mcpwm_timer_);
        if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
        ret = mcpwm_del_timer(mcpwm_timer_);
        if (ret == ESP_OK) mcpwm_timer_ = nullptr;
        else if (first_error == ESP_OK) first_error = ret;
    }

    return first_error;
}

esp_err_t ServoPwm::create_channel_resources(uint8_t index, uint16_t initial_output_us)
{
    if (index >= kServoCount || !states_[index].configured) return ESP_ERR_INVALID_ARG;
    if (mcpwm_comparators_[index] != nullptr || mcpwm_generators_[index] != nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    const uint8_t op_index = channel_to_operator_index(index);
    if (op_index >= kMcpwmOperatorCount || mcpwm_operators_[op_index] == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    const ServoPwmConfig& cfg = states_[index].config;

    mcpwm_comparator_config_t comparator_config = {};
    comparator_config.flags.update_cmp_on_tez = true;
    esp_err_t ret = mcpwm_new_comparator(
        mcpwm_operators_[op_index], &comparator_config, &mcpwm_comparators_[index]);
    if (ret != ESP_OK) return ret;

    mcpwm_generator_config_t generator_config = {};
    generator_config.gen_gpio_num = cfg.gpio;
    ret = mcpwm_new_generator(
        mcpwm_operators_[op_index], &generator_config, &mcpwm_generators_[index]);
    if (ret != ESP_OK) {
        (void)destroy_channel_resources(index);
        return ret;
    }

    ret = mcpwm_comparator_set_compare_value(mcpwm_comparators_[index], initial_output_us);
    if (ret != ESP_OK) {
        (void)destroy_channel_resources(index);
        return ret;
    }

    ret = mcpwm_generator_set_action_on_timer_event(
        mcpwm_generators_[index],
        MCPWM_GEN_TIMER_EVENT_ACTION(
            MCPWM_TIMER_DIRECTION_UP,
            MCPWM_TIMER_EVENT_EMPTY,
            MCPWM_GEN_ACTION_HIGH));
    if (ret != ESP_OK) {
        (void)destroy_channel_resources(index);
        return ret;
    }

    ret = mcpwm_generator_set_action_on_compare_event(
        mcpwm_generators_[index],
        MCPWM_GEN_COMPARE_EVENT_ACTION(
            MCPWM_TIMER_DIRECTION_UP,
            mcpwm_comparators_[index],
            MCPWM_GEN_ACTION_LOW));
    if (ret != ESP_OK) {
        (void)destroy_channel_resources(index);
        return ret;
    }

    ESP_LOGI(TAG, "mcpwm channel S%u gpio=%d op=%u compare=%uus ready",
             static_cast<unsigned>(index), static_cast<int>(cfg.gpio),
             static_cast<unsigned>(op_index), initial_output_us);
    return ESP_OK;
}

esp_err_t ServoPwm::destroy_channel_resources(uint8_t index)
{
    if (index >= kServoCount) return ESP_ERR_INVALID_ARG;
    esp_err_t first_error = ESP_OK;

    if (mcpwm_generators_[index] != nullptr) {
        esp_err_t ret = mcpwm_generator_set_force_level(mcpwm_generators_[index], 0, true);
        if (ret != ESP_OK) first_error = ret;
        ret = mcpwm_del_generator(mcpwm_generators_[index]);
        if (ret == ESP_OK) mcpwm_generators_[index] = nullptr;
        else if (first_error == ESP_OK) first_error = ret;
    }

    if (mcpwm_comparators_[index] != nullptr) {
        const esp_err_t ret = mcpwm_del_comparator(mcpwm_comparators_[index]);
        if (ret == ESP_OK) mcpwm_comparators_[index] = nullptr;
        else if (first_error == ESP_OK) first_error = ret;
    }
    return first_error;
}

esp_err_t ServoPwm::write_output_us(uint8_t index, uint16_t output_us)
{
    if (index >= kServoCount || !states_[index].configured) return ESP_ERR_INVALID_ARG;
    if (!states_[index].enabled || mcpwm_comparators_[index] == nullptr ||
        mcpwm_generators_[index] == nullptr) return ESP_ERR_INVALID_STATE;

    const ServoPwmConfig& cfg = states_[index].config;
    output_us = std::clamp<uint16_t>(output_us, cfg.min_us, cfg.max_us);
    return mcpwm_comparator_set_compare_value(mcpwm_comparators_[index], output_us);
}

esp_err_t ServoPwm::init(const ServoPwmConfig* configs, uint8_t count)
{
    if (initialized_) return ESP_OK;
    if (configs == nullptr || count == 0 || count > kServoCount) return ESP_ERR_INVALID_ARG;

    for (uint8_t i = 0; i < kServoCount; ++i) states_[i] = ServoState{};

    lock_ = xSemaphoreCreateRecursiveMutex();
    if (lock_ == nullptr) return ESP_ERR_NO_MEM;

    for (uint8_t i = 0; i < count; ++i) {
        ServoPwmConfig cfg = configs[i];
        const int index = channel_to_index(cfg.channel);
        if (index < 0 || states_[index].configured) {
            vSemaphoreDelete(lock_);
            lock_ = nullptr;
            return ESP_ERR_INVALID_ARG;
        }

        cfg.min_us = std::clamp<uint16_t>(cfg.min_us, kServoMinUs, kServoMaxUs);
        cfg.max_us = std::clamp<uint16_t>(cfg.max_us, kServoMinUs, kServoMaxUs);
        if (cfg.min_us > cfg.max_us) {
            vSemaphoreDelete(lock_);
            lock_ = nullptr;
            return ESP_ERR_INVALID_ARG;
        }
        cfg.reset_us = std::clamp<uint16_t>(cfg.reset_us, cfg.min_us, cfg.max_us);

        ServoState& s = states_[index];
        s.configured = true;
        s.enabled = false;
        s.config = cfg;
        s.command_us = cfg.reset_us;
        s.output_us = apply_offset_and_clamp(static_cast<uint8_t>(index), cfg.reset_us);
    }

    esp_err_t ret = setup_pwm_backend();
    if (ret != ESP_OK) {
        vSemaphoreDelete(lock_);
        lock_ = nullptr;
        return ret;
    }

    initialized_ = true;
    ESP_LOGI(TAG, "servo_pwm initialized with MCPWM backend, 50Hz, 1us resolution");
    return ESP_OK;
}

esp_err_t ServoPwm::deinit()
{
    if (!initialized_) return ESP_OK;

    esp_err_t ret = ESP_OK;
    {
        LockGuard guard(this);
        if (!guard.locked()) {
            ret = ESP_ERR_TIMEOUT;
        } else {
            const esp_err_t disable_ret = disable_all();
            const esp_err_t teardown_ret = teardown_pwm_backend();
            ret = disable_ret != ESP_OK ? disable_ret : teardown_ret;
            if (ret == ESP_OK) initialized_ = false;
        }
    }

    if (!initialized_ && lock_ != nullptr) {
        vSemaphoreDelete(lock_);
        lock_ = nullptr;
    }
    return ret;
}

esp_err_t ServoPwm::enable(ServoChannel channel, uint16_t start_us)
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;
    if (!initialized_) return ESP_ERR_INVALID_STATE;

    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return ESP_ERR_INVALID_ARG;

    ServoState& s = states_[index];
    if (s.enabled || mcpwm_comparators_[index] != nullptr ||
        mcpwm_generators_[index] != nullptr) return ESP_ERR_INVALID_STATE;

    const uint16_t command_us = clamp_command_us(static_cast<uint8_t>(index), start_us);
    const uint16_t output_us = apply_offset_and_clamp(static_cast<uint8_t>(index), command_us);
    const esp_err_t ret = create_channel_resources(static_cast<uint8_t>(index), output_us);
    if (ret != ESP_OK) return ret;

    s.command_us = command_us;
    s.output_us = output_us;
    s.enabled = true;
    return ESP_OK;
}

esp_err_t ServoPwm::disable(ServoChannel channel)
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;

    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return ESP_ERR_INVALID_ARG;

    ServoState& s = states_[index];
    esp_err_t first_error = destroy_channel_resources(static_cast<uint8_t>(index));
    const gpio_num_t gpio = s.config.gpio;

    esp_err_t ret = gpio_reset_pin(gpio);
    if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    ret = gpio_set_level(gpio, 0);
    if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    ret = gpio_set_direction(gpio, GPIO_MODE_OUTPUT);
    if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    if (first_error != ESP_OK) return first_error;

    s.enabled = false;
    return ESP_OK;
}

esp_err_t ServoPwm::disable_all()
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;

    esp_err_t first_error = ESP_OK;
    for (uint8_t i = 0; i < kServoCount; ++i) {
        if (!states_[i].configured) continue;
        const esp_err_t ret = disable(static_cast<ServoChannel>(i));
        if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    }
    return first_error;
}

bool ServoPwm::is_enabled(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return false;
    const int index = channel_to_index(channel);
    return index >= 0 && states_[index].configured && states_[index].enabled;
}

esp_err_t ServoPwm::write_us_now(ServoChannel channel, uint16_t pulse_us)
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;
    if (!initialized_) return ESP_ERR_INVALID_STATE;

    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return ESP_ERR_INVALID_ARG;
    ServoState& s = states_[index];
    if (!s.enabled) return ESP_ERR_INVALID_STATE;

    const uint16_t command_us = clamp_command_us(static_cast<uint8_t>(index), pulse_us);
    const uint16_t output_us = apply_offset_and_clamp(static_cast<uint8_t>(index), command_us);
    const esp_err_t ret = write_output_us(static_cast<uint8_t>(index), output_us);
    if (ret != ESP_OK) return ret;
    s.command_us = command_us;
    s.output_us = output_us;
    return ESP_OK;
}

esp_err_t ServoPwm::set_config(ServoChannel channel, const ServoPwmConfig& config)
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;

    const int index = channel_to_index(channel);
    if (index < 0) return ESP_ERR_INVALID_ARG;
    if (states_[index].enabled) return ESP_ERR_INVALID_STATE;

    const bool need_recreate = initialized_ &&
        (mcpwm_generators_[index] != nullptr || mcpwm_comparators_[index] != nullptr);
    if (need_recreate) {
        const esp_err_t ret = destroy_channel_resources(static_cast<uint8_t>(index));
        if (ret != ESP_OK) return ret;
    }

    ServoPwmConfig cfg = config;
    cfg.channel = channel;
    cfg.min_us = std::clamp<uint16_t>(cfg.min_us, kServoMinUs, kServoMaxUs);
    cfg.max_us = std::clamp<uint16_t>(cfg.max_us, kServoMinUs, kServoMaxUs);
    if (cfg.min_us > cfg.max_us) return ESP_ERR_INVALID_ARG;
    cfg.reset_us = std::clamp<uint16_t>(cfg.reset_us, cfg.min_us, cfg.max_us);

    ServoState& s = states_[index];
    s.configured = true;
    s.enabled = false;
    s.config = cfg;
    s.command_us = cfg.reset_us;
    s.output_us = apply_offset_and_clamp(static_cast<uint8_t>(index), cfg.reset_us);
    return ESP_OK;
}

ServoPwmConfig ServoPwm::get_config(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return {};
    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return {};
    return states_[index].config;
}

esp_err_t ServoPwm::set_offset(ServoChannel channel, int16_t offset_us)
{
    const LockGuard guard(this);
    if (!guard.locked()) return ESP_ERR_TIMEOUT;
    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return ESP_ERR_INVALID_ARG;

    ServoState& s = states_[index];
    const int unclamped_output = static_cast<int>(s.command_us) + static_cast<int>(offset_us);
    const uint16_t output_us = static_cast<uint16_t>(std::clamp<int>(
        unclamped_output, s.config.min_us, s.config.max_us));

    if (!s.enabled) {
        s.config.offset_us = offset_us;
        s.output_us = output_us;
        return ESP_OK;
    }

    const esp_err_t ret = write_output_us(static_cast<uint8_t>(index), output_us);
    if (ret != ESP_OK) return ret;
    s.config.offset_us = offset_us;
    s.output_us = output_us;
    return ESP_OK;
}

int16_t ServoPwm::get_offset(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return 0;
    const int index = channel_to_index(channel);
    return (index >= 0 && states_[index].configured) ? states_[index].config.offset_us : 0;
}

uint16_t ServoPwm::read_command_us(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return 0;
    const int index = channel_to_index(channel);
    return (index >= 0 && states_[index].configured) ? states_[index].command_us : 0;
}

uint16_t ServoPwm::read_output_us(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return 0;
    const int index = channel_to_index(channel);
    return (index >= 0 && states_[index].configured) ? states_[index].output_us : 0;
}

ServoRuntimeState ServoPwm::get_state(ServoChannel channel) const
{
    const LockGuard guard(this);
    if (!guard.locked()) return {};
    const int index = channel_to_index(channel);
    if (index < 0 || !states_[index].configured) return {};

    const ServoState& s = states_[index];
    ServoRuntimeState state = {};
    state.configured = s.configured;
    state.enabled = s.enabled;
    state.command_us = s.command_us;
    state.output_us = s.output_us;
    state.reset_us = s.config.reset_us;
    state.min_us = s.config.min_us;
    state.max_us = s.config.max_us;
    state.offset_us = s.config.offset_us;
    return state;
}

bool ServoPwm::take_lock(TickType_t timeout_ticks) const
{
    if (lock_ == nullptr) return true;
    return xSemaphoreTakeRecursive(lock_, timeout_ticks) == pdTRUE;
}

void ServoPwm::give_lock() const
{
    if (lock_ != nullptr) xSemaphoreGiveRecursive(lock_);
}

ServoPwm::LockGuard::LockGuard(const ServoPwm* owner, TickType_t timeout_ticks)
    : owner_(owner)
{
    locked_ = (owner_ != nullptr) && owner_->take_lock(timeout_ticks);
}

ServoPwm::LockGuard::~LockGuard()
{
    if (locked_ && owner_ != nullptr) owner_->give_lock();
}

bool ServoPwm::LockGuard::locked() const
{
    return locked_;
}

}  // namespace learm
