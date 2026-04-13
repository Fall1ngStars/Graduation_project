// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/actuator_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "ranger_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "ranger_msgs/msg/detail/actuator_state__struct.h"
#include "ranger_msgs/msg/detail/actuator_state__functions.h"
#include "fastcdr/Cdr.h"

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

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "ranger_msgs/msg/detail/driver_state__functions.h"  // driver
#include "ranger_msgs/msg/detail/motor_state__functions.h"  // motor

// forward declare type support functions
size_t get_serialized_size_ranger_msgs__msg__DriverState(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_ranger_msgs__msg__DriverState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ranger_msgs, msg, DriverState)();
size_t get_serialized_size_ranger_msgs__msg__MotorState(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_ranger_msgs__msg__MotorState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ranger_msgs, msg, MotorState)();


using _ActuatorState__ros_msg_type = ranger_msgs__msg__ActuatorState;

static bool _ActuatorState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ActuatorState__ros_msg_type * ros_message = static_cast<const _ActuatorState__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr << ros_message->id;
  }

  // Field name: motor
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, ranger_msgs, msg, MotorState
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->motor, cdr))
    {
      return false;
    }
  }

  // Field name: driver
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, ranger_msgs, msg, DriverState
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->driver, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _ActuatorState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ActuatorState__ros_msg_type * ros_message = static_cast<_ActuatorState__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr >> ros_message->id;
  }

  // Field name: motor
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, ranger_msgs, msg, MotorState
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->motor))
    {
      return false;
    }
  }

  // Field name: driver
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, ranger_msgs, msg, DriverState
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->driver))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ranger_msgs
size_t get_serialized_size_ranger_msgs__msg__ActuatorState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ActuatorState__ros_msg_type * ros_message = static_cast<const _ActuatorState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name id
  {
    size_t item_size = sizeof(ros_message->id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name motor

  current_alignment += get_serialized_size_ranger_msgs__msg__MotorState(
    &(ros_message->motor), current_alignment);
  // field.name driver

  current_alignment += get_serialized_size_ranger_msgs__msg__DriverState(
    &(ros_message->driver), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _ActuatorState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_ranger_msgs__msg__ActuatorState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ranger_msgs
size_t max_serialized_size_ranger_msgs__msg__ActuatorState(
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

  // member: id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: motor
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_ranger_msgs__msg__MotorState(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: driver
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_ranger_msgs__msg__DriverState(
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
    using DataType = ranger_msgs__msg__ActuatorState;
    is_plain =
      (
      offsetof(DataType, driver) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _ActuatorState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ranger_msgs__msg__ActuatorState(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_ActuatorState = {
  "ranger_msgs::msg",
  "ActuatorState",
  _ActuatorState__cdr_serialize,
  _ActuatorState__cdr_deserialize,
  _ActuatorState__get_serialized_size,
  _ActuatorState__max_serialized_size
};

static rosidl_message_type_support_t _ActuatorState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ActuatorState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ranger_msgs, msg, ActuatorState)() {
  return &_ActuatorState__type_support;
}

#if defined(__cplusplus)
}
#endif
