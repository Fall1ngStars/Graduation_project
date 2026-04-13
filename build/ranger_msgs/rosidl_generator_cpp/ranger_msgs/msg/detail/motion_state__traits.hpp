// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ranger_msgs:msg/MotionState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTION_STATE__TRAITS_HPP_
#define RANGER_MSGS__MSG__DETAIL__MOTION_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ranger_msgs/msg/detail/motion_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace ranger_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MotionState & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: motion_mode
  {
    out << "motion_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.motion_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MotionState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: motion_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motion_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.motion_mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MotionState & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace ranger_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ranger_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ranger_msgs::msg::MotionState & msg,
  std::ostream & out, size_t indentation = 0)
{
  ranger_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ranger_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ranger_msgs::msg::MotionState & msg)
{
  return ranger_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ranger_msgs::msg::MotionState>()
{
  return "ranger_msgs::msg::MotionState";
}

template<>
inline const char * name<ranger_msgs::msg::MotionState>()
{
  return "ranger_msgs/msg/MotionState";
}

template<>
struct has_fixed_size<ranger_msgs::msg::MotionState>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<ranger_msgs::msg::MotionState>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<ranger_msgs::msg::MotionState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RANGER_MSGS__MSG__DETAIL__MOTION_STATE__TRAITS_HPP_
