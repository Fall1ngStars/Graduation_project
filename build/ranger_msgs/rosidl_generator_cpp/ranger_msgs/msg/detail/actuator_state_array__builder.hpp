// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/actuator_state_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_ActuatorStateArray_states
{
public:
  explicit Init_ActuatorStateArray_states(::ranger_msgs::msg::ActuatorStateArray & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::ActuatorStateArray states(::ranger_msgs::msg::ActuatorStateArray::_states_type arg)
  {
    msg_.states = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::ActuatorStateArray msg_;
};

class Init_ActuatorStateArray_header
{
public:
  Init_ActuatorStateArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ActuatorStateArray_states header(::ranger_msgs::msg::ActuatorStateArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ActuatorStateArray_states(msg_);
  }

private:
  ::ranger_msgs::msg::ActuatorStateArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::ActuatorStateArray>()
{
  return ranger_msgs::msg::builder::Init_ActuatorStateArray_header();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__BUILDER_HPP_
