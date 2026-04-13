// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "ranger_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "ranger_msgs/msg/detail/actuator_state__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace ranger_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
cdr_serialize(
  const ranger_msgs::msg::ActuatorState & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ranger_msgs::msg::ActuatorState & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
get_serialized_size(
  const ranger_msgs::msg::ActuatorState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
max_serialized_size_ActuatorState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace ranger_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ranger_msgs, msg, ActuatorState)();

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
