// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/DriverState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/driver_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_DriverState_driver_state
{
public:
  explicit Init_DriverState_driver_state(::ranger_msgs::msg::DriverState & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::DriverState driver_state(::ranger_msgs::msg::DriverState::_driver_state_type arg)
  {
    msg_.driver_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::DriverState msg_;
};

class Init_DriverState_motor_temperature
{
public:
  explicit Init_DriverState_motor_temperature(::ranger_msgs::msg::DriverState & msg)
  : msg_(msg)
  {}
  Init_DriverState_driver_state motor_temperature(::ranger_msgs::msg::DriverState::_motor_temperature_type arg)
  {
    msg_.motor_temperature = std::move(arg);
    return Init_DriverState_driver_state(msg_);
  }

private:
  ::ranger_msgs::msg::DriverState msg_;
};

class Init_DriverState_driver_temperature
{
public:
  explicit Init_DriverState_driver_temperature(::ranger_msgs::msg::DriverState & msg)
  : msg_(msg)
  {}
  Init_DriverState_motor_temperature driver_temperature(::ranger_msgs::msg::DriverState::_driver_temperature_type arg)
  {
    msg_.driver_temperature = std::move(arg);
    return Init_DriverState_motor_temperature(msg_);
  }

private:
  ::ranger_msgs::msg::DriverState msg_;
};

class Init_DriverState_driver_voltage
{
public:
  Init_DriverState_driver_voltage()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DriverState_driver_temperature driver_voltage(::ranger_msgs::msg::DriverState::_driver_voltage_type arg)
  {
    msg_.driver_voltage = std::move(arg);
    return Init_DriverState_driver_temperature(msg_);
  }

private:
  ::ranger_msgs::msg::DriverState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::DriverState>()
{
  return ranger_msgs::msg::builder::Init_DriverState_driver_voltage();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__BUILDER_HPP_
