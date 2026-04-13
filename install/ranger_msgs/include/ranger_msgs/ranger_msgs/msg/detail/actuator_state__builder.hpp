// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/actuator_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_ActuatorState_driver
{
public:
  explicit Init_ActuatorState_driver(::ranger_msgs::msg::ActuatorState & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::ActuatorState driver(::ranger_msgs::msg::ActuatorState::_driver_type arg)
  {
    msg_.driver = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::ActuatorState msg_;
};

class Init_ActuatorState_motor
{
public:
  explicit Init_ActuatorState_motor(::ranger_msgs::msg::ActuatorState & msg)
  : msg_(msg)
  {}
  Init_ActuatorState_driver motor(::ranger_msgs::msg::ActuatorState::_motor_type arg)
  {
    msg_.motor = std::move(arg);
    return Init_ActuatorState_driver(msg_);
  }

private:
  ::ranger_msgs::msg::ActuatorState msg_;
};

class Init_ActuatorState_id
{
public:
  Init_ActuatorState_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ActuatorState_motor id(::ranger_msgs::msg::ActuatorState::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_ActuatorState_motor(msg_);
  }

private:
  ::ranger_msgs::msg::ActuatorState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::ActuatorState>()
{
  return ranger_msgs::msg::builder::Init_ActuatorState_id();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__BUILDER_HPP_
