// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ranger_msgs/msg/detail/actuator_state_array__rosidl_typesupport_introspection_c.h"
#include "ranger_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ranger_msgs/msg/detail/actuator_state_array__functions.h"
#include "ranger_msgs/msg/detail/actuator_state_array__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `states`
#include "ranger_msgs/msg/actuator_state.h"
// Member `states`
#include "ranger_msgs/msg/detail/actuator_state__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ranger_msgs__msg__ActuatorStateArray__init(message_memory);
}

void ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_fini_function(void * message_memory)
{
  ranger_msgs__msg__ActuatorStateArray__fini(message_memory);
}

size_t ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__size_function__ActuatorStateArray__states(
  const void * untyped_member)
{
  const ranger_msgs__msg__ActuatorState__Sequence * member =
    (const ranger_msgs__msg__ActuatorState__Sequence *)(untyped_member);
  return member->size;
}

const void * ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_const_function__ActuatorStateArray__states(
  const void * untyped_member, size_t index)
{
  const ranger_msgs__msg__ActuatorState__Sequence * member =
    (const ranger_msgs__msg__ActuatorState__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_function__ActuatorStateArray__states(
  void * untyped_member, size_t index)
{
  ranger_msgs__msg__ActuatorState__Sequence * member =
    (ranger_msgs__msg__ActuatorState__Sequence *)(untyped_member);
  return &member->data[index];
}

void ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__fetch_function__ActuatorStateArray__states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const ranger_msgs__msg__ActuatorState * item =
    ((const ranger_msgs__msg__ActuatorState *)
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_const_function__ActuatorStateArray__states(untyped_member, index));
  ranger_msgs__msg__ActuatorState * value =
    (ranger_msgs__msg__ActuatorState *)(untyped_value);
  *value = *item;
}

void ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__assign_function__ActuatorStateArray__states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  ranger_msgs__msg__ActuatorState * item =
    ((ranger_msgs__msg__ActuatorState *)
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_function__ActuatorStateArray__states(untyped_member, index));
  const ranger_msgs__msg__ActuatorState * value =
    (const ranger_msgs__msg__ActuatorState *)(untyped_value);
  *item = *value;
}

bool ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__resize_function__ActuatorStateArray__states(
  void * untyped_member, size_t size)
{
  ranger_msgs__msg__ActuatorState__Sequence * member =
    (ranger_msgs__msg__ActuatorState__Sequence *)(untyped_member);
  ranger_msgs__msg__ActuatorState__Sequence__fini(member);
  return ranger_msgs__msg__ActuatorState__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__ActuatorStateArray, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "states",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__ActuatorStateArray, states),  // bytes offset in struct
    NULL,  // default value
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__size_function__ActuatorStateArray__states,  // size() function pointer
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_const_function__ActuatorStateArray__states,  // get_const(index) function pointer
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__get_function__ActuatorStateArray__states,  // get(index) function pointer
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__fetch_function__ActuatorStateArray__states,  // fetch(index, &value) function pointer
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__assign_function__ActuatorStateArray__states,  // assign(index, value) function pointer
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__resize_function__ActuatorStateArray__states  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_members = {
  "ranger_msgs__msg",  // message namespace
  "ActuatorStateArray",  // message name
  2,  // number of fields
  sizeof(ranger_msgs__msg__ActuatorStateArray),
  ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_member_array,  // message members
  ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_init_function,  // function to initialize message memory (memory has to be allocated)
  ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_type_support_handle = {
  0,
  &ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ranger_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, ActuatorStateArray)() {
  ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, ActuatorState)();
  if (!ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_type_support_handle.typesupport_identifier) {
    ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ranger_msgs__msg__ActuatorStateArray__rosidl_typesupport_introspection_c__ActuatorStateArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
