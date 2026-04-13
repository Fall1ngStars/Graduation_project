// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__TRAITS_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ranger_msgs/msg/detail/actuator_state_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'states'
#include "ranger_msgs/msg/detail/actuator_state__traits.hpp"

namespace ranger_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ActuatorStateArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: states
  {
    if (msg.states.size() == 0) {
      out << "states: []";
    } else {
      out << "states: [";
      size_t pending_items = msg.states.size();
      for (auto item : msg.states) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ActuatorStateArray & msg,
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

  // member: states
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.states.size() == 0) {
      out << "states: []\n";
    } else {
      out << "states:\n";
      for (auto item : msg.states) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ActuatorStateArray & msg, bool use_flow_style = false)
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
  const ranger_msgs::msg::ActuatorStateArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  ranger_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ranger_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ranger_msgs::msg::ActuatorStateArray & msg)
{
  return ranger_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ranger_msgs::msg::ActuatorStateArray>()
{
  return "ranger_msgs::msg::ActuatorStateArray";
}

template<>
inline const char * name<ranger_msgs::msg::ActuatorStateArray>()
{
  return "ranger_msgs/msg/ActuatorStateArray";
}

template<>
struct has_fixed_size<ranger_msgs::msg::ActuatorStateArray>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ranger_msgs::msg::ActuatorStateArray>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ranger_msgs::msg::ActuatorStateArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__TRAITS_HPP_
