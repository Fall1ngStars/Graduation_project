// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__TRAITS_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ranger_msgs/msg/detail/actuator_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'motor'
#include "ranger_msgs/msg/detail/motor_state__traits.hpp"
// Member 'driver'
#include "ranger_msgs/msg/detail/driver_state__traits.hpp"

namespace ranger_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ActuatorState & msg,
  std::ostream & out)
{
  out << "{";
  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: motor
  {
    out << "motor: ";
    to_flow_style_yaml(msg.motor, out);
    out << ", ";
  }

  // member: driver
  {
    out << "driver: ";
    to_flow_style_yaml(msg.driver, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ActuatorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: motor
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motor:\n";
    to_block_style_yaml(msg.motor, out, indentation + 2);
  }

  // member: driver
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "driver:\n";
    to_block_style_yaml(msg.driver, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ActuatorState & msg, bool use_flow_style = false)
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
  const ranger_msgs::msg::ActuatorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  ranger_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ranger_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ranger_msgs::msg::ActuatorState & msg)
{
  return ranger_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ranger_msgs::msg::ActuatorState>()
{
  return "ranger_msgs::msg::ActuatorState";
}

template<>
inline const char * name<ranger_msgs::msg::ActuatorState>()
{
  return "ranger_msgs/msg/ActuatorState";
}

template<>
struct has_fixed_size<ranger_msgs::msg::ActuatorState>
  : std::integral_constant<bool, has_fixed_size<ranger_msgs::msg::DriverState>::value && has_fixed_size<ranger_msgs::msg::MotorState>::value> {};

template<>
struct has_bounded_size<ranger_msgs::msg::ActuatorState>
  : std::integral_constant<bool, has_bounded_size<ranger_msgs::msg::DriverState>::value && has_bounded_size<ranger_msgs::msg::MotorState>::value> {};

template<>
struct is_message<ranger_msgs::msg::ActuatorState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__TRAITS_HPP_
