// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/MotorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/motor_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_MotorState_motor_speeds
{
public:
  explicit Init_MotorState_motor_speeds(::ranger_msgs::msg::MotorState & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::MotorState motor_speeds(::ranger_msgs::msg::MotorState::_motor_speeds_type arg)
  {
    msg_.motor_speeds = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::MotorState msg_;
};

class Init_MotorState_motor_angles
{
public:
  explicit Init_MotorState_motor_angles(::ranger_msgs::msg::MotorState & msg)
  : msg_(msg)
  {}
  Init_MotorState_motor_speeds motor_angles(::ranger_msgs::msg::MotorState::_motor_angles_type arg)
  {
    msg_.motor_angles = std::move(arg);
    return Init_MotorState_motor_speeds(msg_);
  }

private:
  ::ranger_msgs::msg::MotorState msg_;
};

class Init_MotorState_pulse_count
{
public:
  explicit Init_MotorState_pulse_count(::ranger_msgs::msg::MotorState & msg)
  : msg_(msg)
  {}
  Init_MotorState_motor_angles pulse_count(::ranger_msgs::msg::MotorState::_pulse_count_type arg)
  {
    msg_.pulse_count = std::move(arg);
    return Init_MotorState_motor_angles(msg_);
  }

private:
  ::ranger_msgs::msg::MotorState msg_;
};

class Init_MotorState_current
{
public:
  explicit Init_MotorState_current(::ranger_msgs::msg::MotorState & msg)
  : msg_(msg)
  {}
  Init_MotorState_pulse_count current(::ranger_msgs::msg::MotorState::_current_type arg)
  {
    msg_.current = std::move(arg);
    return Init_MotorState_pulse_count(msg_);
  }

private:
  ::ranger_msgs::msg::MotorState msg_;
};

class Init_MotorState_rpm
{
public:
  Init_MotorState_rpm()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotorState_current rpm(::ranger_msgs::msg::MotorState::_rpm_type arg)
  {
    msg_.rpm = std::move(arg);
    return Init_MotorState_current(msg_);
  }

private:
  ::ranger_msgs::msg::MotorState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::MotorState>()
{
  return ranger_msgs::msg::builder::Init_MotorState_rpm();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__BUILDER_HPP_
