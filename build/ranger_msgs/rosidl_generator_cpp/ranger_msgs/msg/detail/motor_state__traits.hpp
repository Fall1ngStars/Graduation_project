// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ranger_msgs:msg/MotorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__TRAITS_HPP_
#define RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ranger_msgs/msg/detail/motor_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ranger_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MotorState & msg,
  std::ostream & out)
{
  out << "{";
  // member: rpm
  {
    out << "rpm: ";
    rosidl_generator_traits::value_to_yaml(msg.rpm, out);
    out << ", ";
  }

  // member: current
  {
    out << "current: ";
    rosidl_generator_traits::value_to_yaml(msg.current, out);
    out << ", ";
  }

  // member: pulse_count
  {
    out << "pulse_count: ";
    rosidl_generator_traits::value_to_yaml(msg.pulse_count, out);
    out << ", ";
  }

  // member: motor_angles
  {
    out << "motor_angles: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_angles, out);
    out << ", ";
  }

  // member: motor_speeds
  {
    out << "motor_speeds: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_speeds, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MotorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: rpm
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rpm: ";
    rosidl_generator_traits::value_to_yaml(msg.rpm, out);
    out << "\n";
  }

  // member: current
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current: ";
    rosidl_generator_traits::value_to_yaml(msg.current, out);
    out << "\n";
  }

  // member: pulse_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pulse_count: ";
    rosidl_generator_traits::value_to_yaml(msg.pulse_count, out);
    out << "\n";
  }

  // member: motor_angles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motor_angles: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_angles, out);
    out << "\n";
  }

  // member: motor_speeds
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motor_speeds: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_speeds, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MotorState & msg, bool use_flow_style = false)
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
  const ranger_msgs::msg::MotorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  ranger_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ranger_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ranger_msgs::msg::MotorState & msg)
{
  return ranger_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ranger_msgs::msg::MotorState>()
{
  return "ranger_msgs::msg::MotorState";
}

template<>
inline const char * name<ranger_msgs::msg::MotorState>()
{
  return "ranger_msgs/msg/MotorState";
}

template<>
struct has_fixed_size<ranger_msgs::msg::MotorState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ranger_msgs::msg::MotorState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ranger_msgs::msg::MotorState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__TRAITS_HPP_
