// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/actuator_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "ranger_msgs/msg/detail/actuator_state__struct.hpp"

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
bool cdr_serialize(
  const ranger_msgs::msg::MotorState &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  ranger_msgs::msg::MotorState &);
size_t get_serialized_size(
  const ranger_msgs::msg::MotorState &,
  size_t current_alignment);
size_t
max_serialized_size_MotorState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace ranger_msgs

namespace ranger_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const ranger_msgs::msg::DriverState &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  ranger_msgs::msg::DriverState &);
size_t get_serialized_size(
  const ranger_msgs::msg::DriverState &,
  size_t current_alignment);
size_t
max_serialized_size_DriverState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace ranger_msgs


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
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: id
  cdr << ros_message.id;
  // Member: motor
  ranger_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.motor,
    cdr);
  // Member: driver
  ranger_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.driver,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ranger_msgs::msg::ActuatorState & ros_message)
{
  // Member: id
  cdr >> ros_message.id;

  // Member: motor
  ranger_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.motor);

  // Member: driver
  ranger_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.driver);

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
get_serialized_size(
  const ranger_msgs::msg::ActuatorState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: id
  {
    size_t item_size = sizeof(ros_message.id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: motor

  current_alignment +=
    ranger_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.motor, current_alignment);
  // Member: driver

  current_alignment +=
    ranger_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.driver, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ranger_msgs
max_serialized_size_ActuatorState(
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


  // Member: id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: motor
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        ranger_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_MotorState(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: driver
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        ranger_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_DriverState(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = ranger_msgs::msg::ActuatorState;
    is_plain =
      (
      offsetof(DataType, driver) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _ActuatorState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const ranger_msgs::msg::ActuatorState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _ActuatorState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<ranger_msgs::msg::ActuatorState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _ActuatorState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const ranger_msgs::msg::ActuatorState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _ActuatorState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ActuatorState(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _ActuatorState__callbacks = {
  "ranger_msgs::msg",
  "ActuatorState",
  _ActuatorState__cdr_serialize,
  _ActuatorState__cdr_deserialize,
  _ActuatorState__get_serialized_size,
  _ActuatorState__max_serialized_size
};

static rosidl_message_type_support_t _ActuatorState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_ActuatorState__callbacks,
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
get_message_type_support_handle<ranger_msgs::msg::ActuatorState>()
{
  return &ranger_msgs::msg::typesupport_fastrtps_cpp::_ActuatorState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ranger_msgs, msg, ActuatorState)() {
  return &ranger_msgs::msg::typesupport_fastrtps_cpp::_ActuatorState__handle;
}

#ifdef __cplusplus
}
#endif
