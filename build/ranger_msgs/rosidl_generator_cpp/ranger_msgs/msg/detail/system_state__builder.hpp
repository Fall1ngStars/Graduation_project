// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/SystemState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/system_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_SystemState_motion_mode
{
public:
  explicit Init_SystemState_motion_mode(::ranger_msgs::msg::SystemState & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::SystemState motion_mode(::ranger_msgs::msg::SystemState::_motion_mode_type arg)
  {
    msg_.motion_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

class Init_SystemState_battery_voltage
{
public:
  explicit Init_SystemState_battery_voltage(::ranger_msgs::msg::SystemState & msg)
  : msg_(msg)
  {}
  Init_SystemState_motion_mode battery_voltage(::ranger_msgs::msg::SystemState::_battery_voltage_type arg)
  {
    msg_.battery_voltage = std::move(arg);
    return Init_SystemState_motion_mode(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

class Init_SystemState_error_code
{
public:
  explicit Init_SystemState_error_code(::ranger_msgs::msg::SystemState & msg)
  : msg_(msg)
  {}
  Init_SystemState_battery_voltage error_code(::ranger_msgs::msg::SystemState::_error_code_type arg)
  {
    msg_.error_code = std::move(arg);
    return Init_SystemState_battery_voltage(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

class Init_SystemState_control_mode
{
public:
  explicit Init_SystemState_control_mode(::ranger_msgs::msg::SystemState & msg)
  : msg_(msg)
  {}
  Init_SystemState_error_code control_mode(::ranger_msgs::msg::SystemState::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return Init_SystemState_error_code(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

class Init_SystemState_vehicle_state
{
public:
  explicit Init_SystemState_vehicle_state(::ranger_msgs::msg::SystemState & msg)
  : msg_(msg)
  {}
  Init_SystemState_control_mode vehicle_state(::ranger_msgs::msg::SystemState::_vehicle_state_type arg)
  {
    msg_.vehicle_state = std::move(arg);
    return Init_SystemState_control_mode(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

class Init_SystemState_header
{
public:
  Init_SystemState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SystemState_vehicle_state header(::ranger_msgs::msg::SystemState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SystemState_vehicle_state(msg_);
  }

private:
  ::ranger_msgs::msg::SystemState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::SystemState>()
{
  return ranger_msgs::msg::builder::Init_SystemState_header();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__BUILDER_HPP_
