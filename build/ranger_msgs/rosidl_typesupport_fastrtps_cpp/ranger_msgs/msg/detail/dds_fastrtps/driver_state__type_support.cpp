// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from ranger_msgs:msg/DriverState.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/driver_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "ranger_msgs/msg/detail/driver_state__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace ranger_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
cdr_serialize(
  const ranger_msgs::msg::DriverState & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: driver_voltage
  cdr << ros_message.driver_voltage;
  // Member: driver_temperature
  cdr << ros_message.driver_temperature;
  // Member: motor_temperature
  cdr << ros_message.motor_temperature;
  // Member: driver_state
  cdr << ros_message.driver_state;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ranger_msgs::msg::DriverState & ros_message)
{
  // Member: driver_voltage
  cdr >> ros_message.driver_voltage;

  // Member: driver_temperature
  cdr >> ros_message.driver_temperature;

  // Member: motor_temperature
  cdr >> ros_message.motor_temperature;

  // Member: driver_state
  cdr >> ros_message.driver_state;

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
get_serialized_size(
  const ranger_msgs::msg::DriverState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: driver_voltage
  {
    size_t item_size = sizeof(ros_message.driver_voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: driver_temperature
  {
    size_t item_size = sizeof(ros_message.driver_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: motor_temperature
  {
    size_t item_size = sizeof(ros_message.motor_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: driver_state
  {
    size_t item_size = sizeof(ros_message.driver_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
max_serialized_size_DriverState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: driver_voltage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: driver_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: motor_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: driver_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = ranger_msgs::msg::DriverState;
    is_plain =
      (
      offsetof(DataType, driver_state) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _DriverState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const ranger_msgs::msg::DriverState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _DriverState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<ranger_msgs::msg::DriverState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _DriverState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const ranger_msgs::msg::DriverState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _DriverState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_DriverState(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _DriverState__callbacks = {
  "ranger_msgs::msg",
  "DriverState",
  _DriverState__cdr_serialize,
  _DriverState__cdr_deserialize,
  _DriverState__get_serialized_size,
  _DriverState__max_serialized_size
};

static rosidl_message_type_support_t _DriverState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_DriverState__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace ranger_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_ranger_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<ranger_msgs::msg::DriverState>()
{
  return &ranger_msgs::msg::typesupport_fastrtps_cpp::_DriverState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ranger_msgs, msg, DriverState)() {
  return &ranger_msgs::msg::typesupport_fastrtps_cpp::_DriverState__handle;
}

#ifdef __cplusplus
}
#endif
