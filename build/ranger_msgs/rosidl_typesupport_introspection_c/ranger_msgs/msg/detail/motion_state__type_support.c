// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ranger_msgs:msg/MotionState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ranger_msgs/msg/detail/motion_state__rosidl_typesupport_introspection_c.h"
#include "ranger_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ranger_msgs/msg/detail/motion_state__functions.h"
#include "ranger_msgs/msg/detail/motion_state__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ranger_msgs__msg__MotionState__init(message_memory);
}

void ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_fini_function(void * message_memory)
{
  ranger_msgs__msg__MotionState__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__MotionState, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "motion_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__MotionState, motion_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_members = {
  "ranger_msgs__msg",  // message namespace
  "MotionState",  // message name
  2,  // number of fields
  sizeof(ranger_msgs__msg__MotionState),
  ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_member_array,  // message members
  ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_init_function,  // function to initialize message memory (memory has to be allocated)
  ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_type_support_handle = {
  0,
  &ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ranger_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, MotionState)() {
  ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_type_support_handle.typesupport_identifier) {
    ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ranger_msgs__msg__MotionState__rosidl_typesupport_introspection_c__MotionState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
