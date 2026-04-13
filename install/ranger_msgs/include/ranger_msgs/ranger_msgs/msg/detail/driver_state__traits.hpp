// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ranger_msgs:msg/DriverState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__TRAITS_HPP_
#define RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ranger_msgs/msg/detail/driver_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ranger_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DriverState & msg,
  std::ostream & out)
{
  out << "{";
  // member: driver_voltage
  {
    out << "driver_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_voltage, out);
    out << ", ";
  }

  // member: driver_temperature
  {
    out << "driver_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_temperature, out);
    out << ", ";
  }

  // member: motor_temperature
  {
    out << "motor_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_temperature, out);
    out << ", ";
  }

  // member: driver_state
  {
    out << "driver_state: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DriverState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: driver_voltage
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "driver_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_voltage, out);
    out << "\n";
  }

  // member: driver_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "driver_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_temperature, out);
    out << "\n";
  }

  // member: motor_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motor_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_temperature, out);
    out << "\n";
  }

  // member: driver_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "driver_state: ";
    rosidl_generator_traits::value_to_yaml(msg.driver_state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DriverState & msg, bool use_flow_style = false)
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
  const ranger_msgs::msg::DriverState & msg,
  std::ostream & out, size_t indentation = 0)
{
  ranger_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ranger_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ranger_msgs::msg::DriverState & msg)
{
  return ranger_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ranger_msgs::msg::DriverState>()
{
  return "ranger_msgs::msg::DriverState";
}

template<>
inline const char * name<ranger_msgs::msg::DriverState>()
{
  return "ranger_msgs/msg/DriverState";
}

template<>
struct has_fixed_size<ranger_msgs::msg::DriverState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ranger_msgs::msg::DriverState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ranger_msgs::msg::DriverState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__TRAITS_HPP_
