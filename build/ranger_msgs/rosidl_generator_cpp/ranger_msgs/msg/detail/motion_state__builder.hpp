// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ranger_msgs:msg/MotionState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTION_STATE__BUILDER_HPP_
#define RANGER_MSGS__MSG__DETAIL__MOTION_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ranger_msgs/msg/detail/motion_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ranger_msgs
{

namespace msg
{

namespace builder
{

class Init_MotionState_motion_mode
{
public:
  explicit Init_MotionState_motion_mode(::ranger_msgs::msg::MotionState & msg)
  : msg_(msg)
  {}
  ::ranger_msgs::msg::MotionState motion_mode(::ranger_msgs::msg::MotionState::_motion_mode_type arg)
  {
    msg_.motion_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ranger_msgs::msg::MotionState msg_;
};

class Init_MotionState_header
{
public:
  Init_MotionState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionState_motion_mode header(::ranger_msgs::msg::MotionState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_MotionState_motion_mode(msg_);
  }

private:
  ::ranger_msgs::msg::MotionState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ranger_msgs::msg::MotionState>()
{
  return ranger_msgs::msg::builder::Init_MotionState_header();
}

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__MOTION_STATE__BUILDER_HPP_
